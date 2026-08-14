/* SPDX-License-Identifier: GPL-2.0 */
/*
 * SAURON++ shared definitions (kernel <-> user space)
 * ===================================================
 * Included by both the BPF program (sauron.bpf.c) and the libbpf loader
 * (sauron_loader.c). Keep this header free of kernel-only or libc-only headers
 * so it compiles in both worlds. Only fixed-width integer types are used.
 *
 * Maps to design Section 7 (kernel architecture): the flow_event is the record
 * punted to user space for adaptive scoring; the metric slots back the XDP/TC/
 * LSM counters the dashboard renders.
 */
#ifndef SAURON_H
#define SAURON_H

#ifndef __u8
typedef unsigned char      __u8;
typedef unsigned short     __u16;
typedef unsigned int       __u32;
typedef unsigned long long __u64;
#endif

/* Flow 5-tuple: key for per-flow stats and the fast-path verdict cache. */
struct flow_key {
	__u32 saddr;
	__u32 daddr;
	__u16 sport;
	__u16 dport;
	__u8  proto;
	__u8  _pad[3];
};

/* Per-flow accounting kept in the kernel. */
struct flow_val {
	__u64 packets;
	__u64 bytes;
	__u64 first_ns;
	__u64 last_ns;
	__u32 flags_or;   /* OR of all TCP flags observed on the flow */
	__u32 _pad;
};

/* Record streamed to user space over the ring buffer for scoring. */
struct flow_event {
	__u64 ts_ns;
	__u32 saddr;
	__u32 daddr;
	__u16 sport;
	__u16 dport;
	__u8  proto;      /* IPPROTO_TCP=6, UDP=17, ICMP=1 */
	__u8  flags;      /* SYN=0x02 ACK=0x10 FIN=0x01 RST=0x04 */
	__u16 len;        /* total IP length */
	__u16 hook;       /* 0=XDP ingress, 1=TC egress */
};

/* Per-CPU metric slots. Loader sums across CPUs before reporting. */
enum sauron_metric {
	M_XDP_PKTS = 0,
	M_XDP_BYTES,
	M_XDP_PASS,
	M_XDP_DROP,
	M_TC_PKTS,
	M_TC_BYTES,
	M_LSM_CONNECT,
	M_RB_PUNTS,     /* flow records punted to user space */
	M_RB_MISS,      /* ring buffer full -> record dropped */
	M_PUNT_STARVE,  /* punt suppressed by token-bucket admission (Sec 7.5) */
	M_CG_CONNECT,   /* cgroup/connect4 egress connects observed (Sec 7.6) */
	M_XDP_RATE,     /* fast-path rate-limit verdicts from the cache */
	M_XDP_REDIR,    /* fast-path redirect verdicts from the cache */
	M_METRIC_MAX,
};

/* Fast-path verdict-cache actions (integerized ADE decision, Sec 7.3 / C1). */
#define VERDICT_PASS     0
#define VERDICT_DROP     1
#define VERDICT_RATE     2   /* token-bucket rate limit */
#define VERDICT_REDIRECT 3   /* steer to honeypot ifindex */

/* Quantized decision-cache entry: the ADE result compiled to integers so the
 * kernel can enforce a graduated action with a TTL at line rate, no punt. */
struct verdict_val {
	__u8  action;       /* VERDICT_* */
	__u8  score_band;   /* quantized anomaly score 0..255 (Q0.8) */
	__u16 rate_pps;     /* for VERDICT_RATE: permitted packets/sec */
	__u32 tokens;       /* current token bucket level */
	__u64 refill_ns;    /* last token refill timestamp */
	__u64 expire_ns;    /* auto-expiry (TTL); 0 = permanent */
};

/* Count-Min sketch (heavy-hitter / volumetric detection, Sec 7.4 / C4). */
#define CM_D 4          /* hash rows (depth) */
#define CM_W 1024       /* counters per row (width) */

/* HyperLogLog registers (source/destination fan-out cardinality, Sec 7.4). */
#define HLL_REG 1024    /* 2^10 registers; ~3% cardinality error */

/* Sketch summaries exported to user space (per-CPU; loader merges). */
struct sketch_stats {
	__u64 heavy_hitter_est;  /* max Count-Min estimate observed */
	__u64 cm_updates;        /* total sketch updates */
	__u8  hll_reg[HLL_REG];  /* HLL registers (rho maxima) */
};

/* TCP flag bits we surface. */
#define TCP_SYN 0x02
#define TCP_ACK 0x10
#define TCP_FIN 0x01
#define TCP_RST 0x04

#endif /* SAURON_H */
