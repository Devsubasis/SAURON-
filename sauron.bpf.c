// SPDX-License-Identifier: GPL-2.0
/*
 * SAURON++ in-kernel data path  (design Section 7)
 * ================================================
 * libbpf + CO-RE. Three attach points, one object:
 *
 *   xdp_sauron        SEC("xdp")               ingress fast path / enforcement
 *   tc_sauron         SEC("tc")                egress accounting (clsact)
 *   lsm_socket_connect SEC("lsm/socket_connect") host-level connect monitor
 *
 * Maps
 *   flow_stats   HASH   per-flow packet/byte accounting
 *   verdict      LRU    fast-path verdict cache (the "reflex" path, 7.2)
 *   blocklist    HASH   enforced drop rules (distilled / analyst-approved)
 *   metrics      PERCPU lock-free counters for XDP/TC/LSM
 *   events       RINGBUF flow records punted to user space for scoring
 *
 * Design notes honoured here:
 *   - Bounds checks on every header access (verifier-safe, 7.4).
 *   - Enforcement (blocklist / verdict cache) is checked before any work so a
 *     known-bad source costs almost nothing (7.2).
 *   - Fail-open on *observability* (ring buffer full -> count and continue);
 *     enforcement state lives in maps owned by user space (7.5).
 *
 * Requires: kernel with BTF (CONFIG_DEBUG_INFO_BTF=y). LSM program additionally
 * requires CONFIG_BPF_LSM=y and "bpf" present in the active LSM list; the loader
 * attaches it best-effort and reports availability.
 */
#include "vmlinux.h"
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_endian.h>
#include <bpf/bpf_tracing.h>
#include <bpf/bpf_core_read.h>
#include "sauron.h"

/* vmlinux.h is generated from BTF, which carries *types* but not UAPI #define
 * macros. Define the handful this program needs (guarded so a kernel/header
 * that does provide them still builds). Values are the fixed UAPI constants. */
#ifndef ETH_P_IP
#define ETH_P_IP    0x0800   /* IPv4 EtherType            (linux/if_ether.h) */
#endif
#ifndef ETH_P_IPV6
#define ETH_P_IPV6  0x86DD   /* IPv6 EtherType            (linux/if_ether.h) */
#endif
#ifndef TC_ACT_OK
#define TC_ACT_OK   0        /* pass packet               (linux/pkt_cls.h)  */
#endif
#ifndef TC_ACT_SHOT
#define TC_ACT_SHOT 2        /* drop packet               (linux/pkt_cls.h)  */
#endif

char LICENSE[] SEC("license") = "GPL";

/* ------------------------------------------------------------------ maps */
struct {
	__uint(type, BPF_MAP_TYPE_HASH);
	__type(key, struct flow_key);
	__type(value, struct flow_val);
	__uint(max_entries, 262144);
} flow_stats SEC(".maps");

struct {
	__uint(type, BPF_MAP_TYPE_LRU_HASH);
	__type(key, struct flow_key);
	__type(value, struct verdict_val);   /* quantized ADE decision (7.3/C1) */
	__uint(max_entries, 262144);
} verdict SEC(".maps");

struct {
	__uint(type, BPF_MAP_TYPE_HASH);
	__type(key, __u32);          /* source IPv4 in network byte order */
	__type(value, __u8);
	__uint(max_entries, 65536);
} blocklist SEC(".maps");

struct {
	__uint(type, BPF_MAP_TYPE_PERCPU_ARRAY);
	__type(key, __u32);
	__type(value, __u64);
	__uint(max_entries, M_METRIC_MAX);
} metrics SEC(".maps");

struct {
	__uint(type, BPF_MAP_TYPE_RINGBUF);
	__uint(max_entries, 1 << 22);   /* 4 MiB */
} events SEC(".maps");

/* Count-Min sketch: CM_D rows x CM_W counters, per-CPU (7.4 / C4). */
struct {
	__uint(type, BPF_MAP_TYPE_PERCPU_ARRAY);
	__type(key, __u32);
	__type(value, __u32);
	__uint(max_entries, CM_D * CM_W);
} cmsketch SEC(".maps");

/* HLL registers + heavy-hitter summary, per-CPU (loader merges + estimates). */
struct {
	__uint(type, BPF_MAP_TYPE_PERCPU_ARRAY);
	__type(key, __u32);
	__type(value, struct sketch_stats);
	__uint(max_entries, 1);
} sketch SEC(".maps");

/* Punt-path admission token bucket, per-CPU (7.5). */
struct token_bucket { __u64 tokens; __u64 refill_ns; };
struct {
	__uint(type, BPF_MAP_TYPE_PERCPU_ARRAY);
	__type(key, __u32);
	__type(value, struct token_bucket);
	__uint(max_entries, 1);
} punt_tb SEC(".maps");

/* Punt admission: refill at PUNT_RATE tokens/sec up to PUNT_BURST, spend 1 per
 * punt. Prevents ring-buffer floods under attack (graceful observability). */
#define PUNT_RATE  20000ULL
#define PUNT_BURST 40000ULL

/* Sampling: punt 1/N flow records to user space to bound overhead at line rate.
 * User space still sees every *enforced* drop via metrics; scoring only needs a
 * representative sample (design 8, NFR ingestion). Compile-time so the hot path
 * carries no extra map lookup: build.sh passes -DSAMPLE_N=<n>. */
#ifndef SAMPLE_N
#define SAMPLE_N 1
#endif

/* ------------------------------------------------------------- helpers */
static __always_inline void metric_add(__u32 slot, __u64 v)
{
	__u64 *p = bpf_map_lookup_elem(&metrics, &slot);
	if (p)
		__sync_fetch_and_add(p, v);
}

/* Cheap 32-bit mix (Fibonacci hashing) with a per-row seed. */
static __always_inline __u32 mix32(__u32 x, __u32 seed)
{
	x ^= seed;
	x *= 0x9e3779b1U;
	x ^= x >> 15;
	x *= 0x85ebca77U;
	x ^= x >> 13;
	return x;
}

/* Count-Min update; returns the (min) estimated count for the key. */
static __always_inline __u32 cm_update(__u32 saddr)
{
	__u32 est = 0xffffffffU;
	__u32 seeds[CM_D];
	seeds[0] = 0x00000001U; seeds[1] = 0x9e3779b1U;
	seeds[2] = 0x85ebca77U; seeds[3] = 0xc2b2ae3dU;
#pragma unroll
	for (int d = 0; d < CM_D; d++) {
		__u32 idx = d * CM_W + (mix32(saddr, seeds[d]) & (CM_W - 1));
		__u32 *c = bpf_map_lookup_elem(&cmsketch, &idx);
		if (c) {
			__u32 v = *c + 1;
			*c = v;
			if (v < est)
				est = v;
		}
	}
	return est == 0xffffffffU ? 0 : est;
}

/* Count leading zeros of a 32-bit word. The BPF backend cannot lower the
 * __builtin_clz/CTLZ opcode, so we use a branchless binary search (no loop,
 * verifier-safe). Returns 32 for a zero input. */
static __always_inline __u8 clz32(__u32 x)
{
	__u8 n = 0;
	if (x == 0)
		return 32;
	if ((x & 0xFFFF0000u) == 0) { n += 16; x <<= 16; }
	if ((x & 0xFF000000u) == 0) { n += 8;  x <<= 8;  }
	if ((x & 0xF0000000u) == 0) { n += 4;  x <<= 4;  }
	if ((x & 0xC0000000u) == 0) { n += 2;  x <<= 2;  }
	if ((x & 0x80000000u) == 0) { n += 1; }
	return n;
}

/* HLL update on (saddr,dport); tracks per-register rho maxima + heavy hitter. */
static __always_inline void sketch_update(__u32 saddr, __u16 dport, __u32 cm_est)
{
	__u32 zero = 0;
	struct sketch_stats *s = bpf_map_lookup_elem(&sketch, &zero);
	if (!s)
		return;
	__u32 h = mix32(saddr ^ ((__u32)dport << 16), 0x5bd1e995U);
	__u32 idx = h & (HLL_REG - 1);
	__u32 w = (h >> 10) | 0x1u;          /* ensure nonzero for rank */
	__u8 rho = (__u8)(clz32(w) + 1);
	if (idx < HLL_REG && rho > s->hll_reg[idx])
		s->hll_reg[idx] = rho;
	s->cm_updates++;
	if (cm_est > s->heavy_hitter_est)
		s->heavy_hitter_est = cm_est;
}

/* Token-bucket admission for the punt path. Returns 1 if a punt is allowed. */
static __always_inline int punt_admit(__u64 now)
{
	__u32 zero = 0;
	struct token_bucket *tb = bpf_map_lookup_elem(&punt_tb, &zero);
	if (!tb)
		return 1;
	if (tb->refill_ns == 0)
		tb->refill_ns = now, tb->tokens = PUNT_BURST;
	__u64 elapsed = now - tb->refill_ns;
	if (elapsed > 0) {
		__u64 add = (elapsed * PUNT_RATE) / 1000000000ULL;
		if (add) {
			__u64 t = tb->tokens + add;
			tb->tokens = t > PUNT_BURST ? PUNT_BURST : t;
			tb->refill_ns = now;
		}
	}
	if (tb->tokens > 0) {
		tb->tokens--;
		return 1;
	}
	return 0;
}

/* ------------------------------------------------------------- XDP ingress */
SEC("xdp")
int xdp_sauron(struct xdp_md *ctx)
{
	void *data     = (void *)(long)ctx->data;
	void *data_end = (void *)(long)ctx->data_end;

	struct ethhdr *eth = data;
	if ((void *)(eth + 1) > data_end)
		return XDP_PASS;
	if (eth->h_proto != bpf_htons(ETH_P_IP))
		return XDP_PASS;                     /* IPv4 fast path only */

	struct iphdr *ip = (void *)(eth + 1);
	if ((void *)(ip + 1) > data_end)
		return XDP_PASS;

	__u16 tot_len = bpf_ntohs(ip->tot_len);
	metric_add(M_XDP_PKTS, 1);
	metric_add(M_XDP_BYTES, tot_len);

	/* (1) enforced blocklist: cheapest possible path for known-bad source */
	__u32 saddr = ip->saddr;
	__u8 *blk = bpf_map_lookup_elem(&blocklist, &saddr);
	if (blk && *blk) {
		metric_add(M_XDP_DROP, 1);
		return XDP_DROP;
	}

	/* parse L4 (bounded by ihl) */
	__u32 ihl = ip->ihl * 4;
	if (ihl < sizeof(struct iphdr))
		return XDP_PASS;
	void *l4 = (void *)ip + ihl;

	struct flow_key key = {};
	key.saddr = ip->saddr;
	key.daddr = ip->daddr;
	key.proto = ip->protocol;

	__u16 sport = 0, dport = 0;
	__u8 tflags = 0;

	if (ip->protocol == IPPROTO_TCP) {
		struct tcphdr *tcp = l4;
		if ((void *)(tcp + 1) > data_end)
			return XDP_PASS;
		sport = tcp->source;
		dport = tcp->dest;
		tflags = (tcp->syn ? TCP_SYN : 0) | (tcp->ack ? TCP_ACK : 0) |
			 (tcp->fin ? TCP_FIN : 0) | (tcp->rst ? TCP_RST : 0);
	} else if (ip->protocol == IPPROTO_UDP) {
		struct udphdr *udp = l4;
		if ((void *)(udp + 1) > data_end)
			return XDP_PASS;
		sport = udp->source;
		dport = udp->dest;
	}
	key.sport = sport;
	key.dport = dport;

	/* (2) fast-path quantized verdict cache (reflex path, 7.2/7.3). Enforces a
	 * graduated integer action with a TTL entirely in-kernel — no punt needed
	 * for a flow the ADE has already decided. */
	__u64 now = bpf_ktime_get_ns();
	struct verdict_val *vv = bpf_map_lookup_elem(&verdict, &key);
	if (vv && (vv->expire_ns == 0 || now < vv->expire_ns)) {
		if (vv->action == VERDICT_DROP) {
			metric_add(M_XDP_DROP, 1);
			return XDP_DROP;
		} else if (vv->action == VERDICT_RATE) {
			__u64 add = ((now - vv->refill_ns) * vv->rate_pps) / 1000000000ULL;
			if (add) {
				__u32 t = vv->tokens + (add > 0xffffffffULL ? 0xffffffffU : (__u32)add);
				vv->tokens = t;
				vv->refill_ns = now;
			}
			if (vv->tokens == 0) {
				metric_add(M_XDP_RATE, 1);
				return XDP_DROP;
			}
			vv->tokens--;
		} else if (vv->action == VERDICT_REDIRECT) {
			metric_add(M_XDP_REDIR, 1);
			/* honeypot steering via devmap is wired in the full build */
		}
	}

	/* (3) per-flow accounting + in-kernel sketches (heavy hitter + fan-out) */
	struct flow_val *fv = bpf_map_lookup_elem(&flow_stats, &key);
	if (fv) {
		__sync_fetch_and_add(&fv->packets, 1);
		__sync_fetch_and_add(&fv->bytes, tot_len);
		fv->last_ns = now;
		fv->flags_or |= tflags;
	} else {
		struct flow_val nv = {};
		nv.packets = 1;
		nv.bytes = tot_len;
		nv.first_ns = now;
		nv.last_ns = now;
		nv.flags_or = tflags;
		bpf_map_update_elem(&flow_stats, &key, &nv, BPF_ANY);
	}

	/* Count-Min heavy-hitter estimate + HLL destination fan-out cardinality.
	 * These give volumetric/scan signal to user space without per-flow state. */
	__u32 hh = cm_update(saddr);
	sketch_update(saddr, dport, hh);

	/* (4) punt a sampled flow record to user space for adaptive scoring, gated
	 * by token-bucket admission so a flood cannot overwhelm the ring buffer. */
	if (SAMPLE_N <= 1 || (bpf_get_prandom_u32() % SAMPLE_N) == 0) {
		if (!punt_admit(now)) {
			metric_add(M_PUNT_STARVE, 1);
		} else {
			struct flow_event *e =
				bpf_ringbuf_reserve(&events, sizeof(*e), 0);
			if (e) {
				e->ts_ns = now;
				e->saddr = ip->saddr;
				e->daddr = ip->daddr;
				e->sport = sport;
				e->dport = dport;
				e->proto = ip->protocol;
				e->flags = tflags;
				e->len = tot_len;
				e->hook = 0;
				bpf_ringbuf_submit(e, 0);
				metric_add(M_RB_PUNTS, 1);
			} else {
				metric_add(M_RB_MISS, 1);
			}
		}
	}

	metric_add(M_XDP_PASS, 1);
	return XDP_PASS;
}

/* ------------------------------------------------------------- TC egress */
/* Attached to the clsact egress hook by the loader. Direct packet access on
 * __sk_buff for lightweight, verifier-safe egress accounting (design 7.3). */
SEC("tc")
int tc_sauron(struct __sk_buff *skb)
{
	void *data     = (void *)(long)skb->data;
	void *data_end = (void *)(long)skb->data_end;

	struct ethhdr *eth = data;
	if ((void *)(eth + 1) > data_end)
		return TC_ACT_OK;

	metric_add(M_TC_PKTS, 1);
	metric_add(M_TC_BYTES, skb->len);

	if (eth->h_proto == bpf_htons(ETH_P_IP)) {
		struct iphdr *ip = (void *)(eth + 1);
		if ((void *)(ip + 1) <= data_end) {
			struct flow_event *e =
				bpf_ringbuf_reserve(&events, sizeof(*e), 0);
			if (e) {
				e->ts_ns = bpf_ktime_get_ns();
				e->saddr = ip->saddr;
				e->daddr = ip->daddr;
				e->sport = 0;
				e->dport = 0;
				e->proto = ip->protocol;
				e->flags = 0;
				e->len = bpf_ntohs(ip->tot_len);
				e->hook = 1;
				/* egress records are informational; submit rarely */
				if ((bpf_get_prandom_u32() & 0x1f) == 0)
					bpf_ringbuf_submit(e, 0);
				else
					bpf_ringbuf_discard(e, 0);
			}
		}
	}
	return TC_ACT_OK;
}

/* ------------------------------------------------------------- LSM hook */
/* Host-level visibility: count outbound connect() attempts. Requires BPF LSM.
 * Never blocks here (returns 0 = allow); enforcement stays in the XDP path so
 * policy is centralised. Returning the incoming ret preserves the LSM chain. */
/* ------------------------------------------------------- cgroup connect4 */
/* Socket/cgroup-layer hook (design 7.1 fourth stage, 7.6 cloud-native profile):
 * observes outbound IPv4 connect() at the cgroup boundary — the natural place to
 * enforce egress policy for a container/pod. Consults the same blocklist so a
 * quarantined destination cannot be reached even from a compromised workload.
 * Returns 1 (allow) by default; 0 would block the connect. */
SEC("cgroup/connect4")
int cg_connect4(struct bpf_sock_addr *ctx)
{
	metric_add(M_CG_CONNECT, 1);
	__u32 daddr = ctx->user_ip4;                 /* network byte order */
	__u8 *blk = bpf_map_lookup_elem(&blocklist, &daddr);
	if (blk && *blk)
		return 0;                            /* deny egress to blocked dst */
	return 1;
}
