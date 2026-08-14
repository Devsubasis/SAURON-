// SPDX-License-Identifier: GPL-2.0
/*
 * SAURON++ libbpf loader / kernel bridge  (design Section 8)
 * =========================================================
 * Privileged userspace component. Loads the CO-RE object, attaches all three
 * programs, and bridges the kernel to the Python analytics backend over pipes:
 *
 *   stdout : newline-delimited JSON, one object per line
 *            {"t":"flow", ...}     a sampled flow record (for scoring)
 *            {"t":"metrics", ...}  summed XDP/TC/LSM counters (~4 Hz)
 *            {"t":"status", ...}   attach results at startup
 *   stdin  : text commands, one per line
 *            BLOCK <ipv4>          add source to the kernel blocklist map
 *            UNBLOCK <ipv4>        remove source
 *
 * This keeps BPF loading in idiomatic libbpf/CO-RE C while Python owns the
 * adaptive engine and the WebSocket server. Build with scripts/build.sh.
 *
 * Usage:  sudo ./sauron_loader <ifindex-or-ifname> [path/to/sauron.bpf.o]
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <unistd.h>
#include <signal.h>
#include <net/if.h>
#include <arpa/inet.h>
#include <poll.h>
#include <time.h>
#include <math.h>
#include <bpf/libbpf.h>
#include <bpf/bpf.h>
#include "sauron.h"

/* Generic/SKB-mode XDP flag (linux/if_link.h); define if the headers omit it.
 * SKB mode rides the network stack above the driver, so it attaches on virtual
 * NICs (WSL/Hyper-V netvsc, some cloud netdevs) where native XDP is refused. */
#ifndef XDP_FLAGS_SKB_MODE
#define XDP_FLAGS_SKB_MODE (1U << 1)
#endif

static volatile sig_atomic_t exiting = 0;
static void on_sigint(int sig) { (void)sig; exiting = 1; }

static struct bpf_tc_hook  tc_hook;
static int tc_attached = 0;
static int xdp_skb_mode = 0;
static int ifindex = 0;
static struct bpf_link *xdp_link = NULL;
static struct bpf_link *lsm_link = NULL;

static int libbpf_quiet(enum libbpf_print_level lvl, const char *fmt, va_list ap)
{
	if (lvl == LIBBPF_DEBUG)
		return 0;
	return vfprintf(stderr, fmt, ap);
}

static __u64 now_ms(void)
{
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return (__u64)ts.tv_sec * 1000 + ts.tv_nsec / 1000000;
}

/* ---- ring buffer callback: emit one flow JSON line ---- */
static int on_event(void *ctx, void *data, size_t sz)
{
	(void)ctx;
	if (sz < sizeof(struct flow_event))
		return 0;
	const struct flow_event *e = data;
	struct in_addr s = { .s_addr = e->saddr };
	struct in_addr d = { .s_addr = e->daddr };
	char sbuf[INET_ADDRSTRLEN], dbuf[INET_ADDRSTRLEN];
	inet_ntop(AF_INET, &s, sbuf, sizeof(sbuf));
	inet_ntop(AF_INET, &d, dbuf, sizeof(dbuf));
	printf("{\"t\":\"flow\",\"src\":\"%s\",\"dst\":\"%s\",\"sport\":%u,"
	       "\"dport\":%u,\"proto\":%u,\"flags\":%u,\"len\":%u,\"hook\":%u}\n",
	       sbuf, dbuf, ntohs(e->sport), ntohs(e->dport),
	       e->proto, e->flags, e->len, e->hook);
	return 0;
}

/* ---- read + sum a per-CPU metric slot ---- */
static __u64 metric_sum(int fd, __u32 slot, __u64 *percpu, int ncpu)
{
	if (bpf_map_lookup_elem(fd, &slot, percpu) != 0)
		return 0;
	__u64 total = 0;
	for (int i = 0; i < ncpu; i++)
		total += percpu[i];
	return total;
}

static int sketch_fd = -1;   /* set in main() */

/* HyperLogLog cardinality estimate from merged registers. */
static double hll_estimate(const __u8 *reg, int m)
{
	double sum = 0.0;
	int zeros = 0;
	for (int j = 0; j < m; j++) {
		sum += 1.0 / (double)(1ULL << reg[j]);
		if (reg[j] == 0)
			zeros++;
	}
	double alpha = 0.7213 / (1.0 + 1.079 / (double)m);
	double est = alpha * (double)m * (double)m / sum;
	if (est <= 2.5 * m && zeros > 0)           /* small-range correction */
		est = (double)m * log((double)m / (double)zeros);
	return est;
}

static void emit_metrics(int fd, __u64 *percpu, int ncpu)
{
	__u64 m[M_METRIC_MAX];
	for (int s = 0; s < M_METRIC_MAX; s++)
		m[s] = metric_sum(fd, s, percpu, ncpu);

	/* merge per-CPU sketch state: max HLL register + max heavy-hitter estimate */
	unsigned long long heavy = 0;
	double cardinality = 0.0;
	if (sketch_fd >= 0) {
		int vsz = sizeof(struct sketch_stats);
		struct sketch_stats *ss = calloc(ncpu, vsz);
		__u32 zero = 0;
		if (ss && bpf_map_lookup_elem(sketch_fd, &zero, ss) == 0) {
			__u8 merged[HLL_REG];
			for (int j = 0; j < HLL_REG; j++)
				merged[j] = 0;
			for (int c = 0; c < ncpu; c++) {
				if (ss[c].heavy_hitter_est > heavy)
					heavy = ss[c].heavy_hitter_est;
				for (int j = 0; j < HLL_REG; j++)
					if (ss[c].hll_reg[j] > merged[j])
						merged[j] = ss[c].hll_reg[j];
			}
			cardinality = hll_estimate(merged, HLL_REG);
		}
		free(ss);
	}

	printf("{\"t\":\"metrics\",\"xdp_pkts\":%llu,\"xdp_bytes\":%llu,"
	       "\"xdp_pass\":%llu,\"xdp_drop\":%llu,\"tc_pkts\":%llu,"
	       "\"tc_bytes\":%llu,\"lsm_connect\":%llu,\"rb_punts\":%llu,"
	       "\"rb_miss\":%llu,\"punt_starve\":%llu,\"cg_connect\":%llu,"
	       "\"xdp_rate\":%llu,\"xdp_redir\":%llu,"
	       "\"heavy_hitter\":%llu,\"fanout_cardinality\":%.0f}\n",
	       (unsigned long long)m[M_XDP_PKTS], (unsigned long long)m[M_XDP_BYTES],
	       (unsigned long long)m[M_XDP_PASS], (unsigned long long)m[M_XDP_DROP],
	       (unsigned long long)m[M_TC_PKTS], (unsigned long long)m[M_TC_BYTES],
	       (unsigned long long)m[M_LSM_CONNECT], (unsigned long long)m[M_RB_PUNTS],
	       (unsigned long long)m[M_RB_MISS], (unsigned long long)m[M_PUNT_STARVE],
	       (unsigned long long)m[M_CG_CONNECT], (unsigned long long)m[M_XDP_RATE],
	       (unsigned long long)m[M_XDP_REDIR], heavy, cardinality);
	fflush(stdout);
}

/* ---- apply a stdin command line ---- */
static void handle_command(int blk_fd, char *line)
{
	char cmd[16], ip[64];
	if (sscanf(line, "%15s %63s", cmd, ip) != 2)
		return;
	struct in_addr a;
	if (inet_pton(AF_INET, ip, &a) != 1)
		return;
	__u32 key = a.s_addr;
	if (strcmp(cmd, "BLOCK") == 0) {
		__u8 one = VERDICT_DROP;
		bpf_map_update_elem(blk_fd, &key, &one, BPF_ANY);
		fprintf(stderr, "[loader] BLOCK %s\n", ip);
	} else if (strcmp(cmd, "UNBLOCK") == 0) {
		bpf_map_delete_elem(blk_fd, &key);
		fprintf(stderr, "[loader] UNBLOCK %s\n", ip);
	}
}

static void cleanup(void)
{
	if (xdp_link) bpf_link__destroy(xdp_link);
	else if (xdp_skb_mode) bpf_xdp_detach(ifindex, XDP_FLAGS_SKB_MODE, NULL);
	if (lsm_link) bpf_link__destroy(lsm_link);
	if (tc_attached) {
		tc_hook.attach_point = BPF_TC_EGRESS;
		bpf_tc_hook_destroy(&tc_hook);
	}
}

int main(int argc, char **argv)
{
	if (argc < 2) {
		fprintf(stderr, "usage: %s <ifname|ifindex> [sauron.bpf.o]\n", argv[0]);
		return 1;
	}
	const char *obj_path = (argc >= 3) ? argv[2] : "sauron.bpf.o";

	ifindex = if_nametoindex(argv[1]);
	if (ifindex == 0)
		ifindex = atoi(argv[1]);      /* accept a raw index too */
	if (ifindex == 0) {
		fprintf(stderr, "[loader] invalid interface: %s\n", argv[1]);
		return 1;
	}

	libbpf_set_print(libbpf_quiet);
	signal(SIGINT, on_sigint);
	signal(SIGTERM, on_sigint);

	struct bpf_object *obj = bpf_object__open_file(obj_path, NULL);
	if (!obj || libbpf_get_error(obj)) {
		fprintf(stderr, "[loader] open %s failed: %s\n", obj_path,
			strerror(errno));
		return 1;
	}
	if (bpf_object__load(obj)) {
		fprintf(stderr, "[loader] load failed: %s (need root + BTF kernel)\n",
			strerror(errno));
		bpf_object__close(obj);
		return 1;
	}

	struct bpf_program *xdp_prog = bpf_object__find_program_by_name(obj, "xdp_sauron");
	struct bpf_program *tc_prog  = bpf_object__find_program_by_name(obj, "tc_sauron");
	struct bpf_program *lsm_prog = bpf_object__find_program_by_name(obj, "lsm_socket_connect");
	struct bpf_map *events_map   = bpf_object__find_map_by_name(obj, "events");
	struct bpf_map *metrics_map  = bpf_object__find_map_by_name(obj, "metrics");
	struct bpf_map *blk_map      = bpf_object__find_map_by_name(obj, "blocklist");
	if (!xdp_prog || !events_map || !metrics_map || !blk_map) {
		fprintf(stderr, "[loader] missing program/map in object\n");
		bpf_object__close(obj);
		return 1;
	}

	/* ---- attach XDP ingress ---- */
	int xdp_ok = 0;
	xdp_link = bpf_program__attach_xdp(xdp_prog, ifindex);
	if (!xdp_link || libbpf_get_error(xdp_link)) {
		xdp_link = NULL;
		/* Native/driver XDP not supported (common on virtual NICs such as
		 * WSL/Hyper-V netvsc). Retry in generic/SKB mode. */
		if (bpf_xdp_attach(ifindex, bpf_program__fd(xdp_prog),
				   XDP_FLAGS_SKB_MODE, NULL) == 0) {
			xdp_skb_mode = 1;
			xdp_ok = 1;
			fprintf(stderr, "[loader] XDP attached in generic/SKB mode "
					"on ifindex %d\n", ifindex);
		} else {
			fprintf(stderr, "[loader] XDP attach failed on ifindex %d "
					"(native and SKB modes)\n", ifindex);
		}
	} else {
		xdp_ok = 1;
	}

	/* ---- attach TC egress via clsact ---- */
	int tc_ok = 0;
	if (tc_prog) {
		memset(&tc_hook, 0, sizeof(tc_hook));
		tc_hook.sz = sizeof(tc_hook);
		tc_hook.ifindex = ifindex;
		tc_hook.attach_point = BPF_TC_EGRESS;
		int err = bpf_tc_hook_create(&tc_hook);
		if (err == 0 || err == -EEXIST) {
			tc_attached = 1;
			struct bpf_tc_opts topts;
			memset(&topts, 0, sizeof(topts));
			topts.sz = sizeof(topts);
			topts.prog_fd = bpf_program__fd(tc_prog);
			if (bpf_tc_attach(&tc_hook, &topts) == 0)
				tc_ok = 1;
		}
	}

	/* ---- attach LSM (best-effort; needs CONFIG_BPF_LSM) ---- */
	int lsm_ok = 0;
	if (lsm_prog) {
		lsm_link = bpf_program__attach_lsm(lsm_prog);
		if (lsm_link && !libbpf_get_error(lsm_link))
			lsm_ok = 1;
		else
			lsm_link = NULL;
	}

	printf("{\"t\":\"status\",\"xdp\":%d,\"tc\":%d,\"lsm\":%d,\"ifindex\":%d}\n",
	       xdp_ok, tc_ok, lsm_ok, ifindex);
	fflush(stdout);
	fprintf(stderr, "[loader] attached xdp=%d tc=%d lsm=%d on ifindex %d\n",
		xdp_ok, tc_ok, lsm_ok, ifindex);

	if (!xdp_ok) { cleanup(); bpf_object__close(obj); return 1; }

	/* ---- ring buffer + poll loop ---- */
	struct ring_buffer *rb = ring_buffer__new(bpf_map__fd(events_map),
						  on_event, NULL, NULL);
	if (!rb) {
		fprintf(stderr, "[loader] ring_buffer__new failed\n");
		cleanup(); bpf_object__close(obj); return 1;
	}

	int ncpu = libbpf_num_possible_cpus();
	if (ncpu < 1) ncpu = 1;
	__u64 *percpu = calloc(ncpu, sizeof(__u64));
	int metrics_fd = bpf_map__fd(metrics_map);
	struct bpf_map *sketch_map = bpf_object__find_map_by_name(obj, "sketch");
	if (sketch_map) sketch_fd = bpf_map__fd(sketch_map);
	int blk_fd = bpf_map__fd(blk_map);

	/* line-buffered stdin for commands */
	char inbuf[128];
	struct pollfd pfd = { .fd = 0, .events = POLLIN };
	__u64 last_metrics = 0;

	while (!exiting) {
		ring_buffer__poll(rb, 100 /* ms */);

		/* drain any pending stdin commands (non-blocking) */
		while (poll(&pfd, 1, 0) > 0 && (pfd.revents & POLLIN)) {
			if (!fgets(inbuf, sizeof(inbuf), stdin)) { exiting = 1; break; }
			handle_command(blk_fd, inbuf);
		}

		__u64 t = now_ms();
		if (t - last_metrics >= 250) {
			last_metrics = t;
			emit_metrics(metrics_fd, percpu, ncpu);
		}
	}

	fprintf(stderr, "[loader] shutting down\n");
	free(percpu);
	ring_buffer__free(rb);
	cleanup();
	bpf_object__close(obj);
	return 0;
}
