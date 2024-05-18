#include <linux/bpf.h>
#include <linux/if_ether.h>
#include <linux/tcp.h>
#include <linux/udp.h>
#include <linux/in.h>
#include <linux/ip.h>

#define MAX_TREE_DEPTH  15
#define RANDOM_FOREST_TREE_NUM  51
#define TREE_LEAF -1
#define FEATURE_VEC_LENGTH 17
#define flow_timeout  15000
#define packet_cnt  0
#define tcp_cnt  1
#define udp_cnt  2
#define flow_cnt  3

struct STATISTIC {
    u32 n;
    u32 dev;
    s32 m1;
    u32 m2;
    u32 sum;
    u32 min;
    u32 max;
};

struct FLOW_KEY {
    u8 protocol;
    u32 src;
    u32 dest;
    u32 src_port;
    u32 dest_port;
};

struct FLOW_FEATURE_NODE {
    u8 protocol;

    u8 syn;

    u8 fin;

    u8 rst;

    u8 psh;

    u8 urg;

    u16 win;

    u64 flow_start_time;
    u64 flow_end_time;
    u64 active_start_time;
    u64 active_end_time;

    struct STATISTIC packet_length;

    struct STATISTIC iat;
};

struct PACKET_INFO {
    struct FLOW_KEY *flow_key;
    u8 fin, rst, syn, psh, urg;
    u16 win;
    u64 current_time;
    u32 payload;
};

BPF_TABLE("lru_hash", struct FLOW_KEY,  struct FLOW_FEATURE_NODE, flow_table,  100000);
BPF_TABLE("lru_hash", struct FLOW_KEY,  u32 , exception_table,  10000000);
BPF_PERCPU_ARRAY(statistic, u32, 4);

void static increase(struct STATISTIC *statistic, u32 d) {
    if (statistic->n == 0) {
        statistic->m1 = statistic->m2 = 0;
        statistic->min = statistic->max = d;
    }
    statistic->n++;
    statistic->sum += d;
    if (d < statistic->min) statistic->min = d;
    if (d > statistic->max) statistic->max = d;
    if (d > statistic->m1) {
        statistic->dev = d - statistic->m1;
        statistic->m1 += statistic->dev / statistic->n;
    } else {
        statistic->dev = statistic->m1 - d;
        statistic->m1 -= statistic->dev / statistic->n;
    }
    statistic->m2 += (statistic->n - 1) * statistic->dev * statistic->dev / statistic->n;
}

void static analysis(struct FLOW_FEATURE_NODE *flow, struct FLOW_KEY *flow_key) {
    if (flow->flow_end_time - flow->flow_start_time <= 0 || flow->packet_length.n <= 0) return;

    u64 feature_vec[FEATURE_VEC_LENGTH];

    feature_vec[0] = flow->protocol;

    feature_vec[1] = flow->flow_end_time - flow->flow_start_time;

    feature_vec[2] = flow->packet_length.sum * 1000 / (flow->flow_end_time - flow->flow_start_time);

    feature_vec[3] = flow->packet_length.n * 1000 / (flow->flow_end_time - flow->flow_start_time);

    feature_vec[4] = flow->iat.sum / flow->iat.n;

    feature_vec[5] = flow->iat.m2 / (flow->iat.n - 1);

    feature_vec[6] = flow->iat.max;

    feature_vec[7] = flow->iat.min;

    feature_vec[8] = flow->packet_length.n;

    feature_vec[9] = flow->packet_length.min;

    feature_vec[10] = flow->packet_length.max;

    feature_vec[11] = flow->packet_length.sum / flow->packet_length.n;

    feature_vec[12] = flow->packet_length.m2 / (flow->packet_length.n - 1);

    feature_vec[13] = flow->fin;

    feature_vec[14] = flow->syn;

    feature_vec[15] = flow->rst;

    feature_vec[16] = flow->psh;


    u32 positive = 0;
    u32 idx = 0;
    for (int i = 0; i < RANDOM_FOREST_TREE_NUM; i++) {
        u32 current_node = idx;
        for (int j = 0; j < MAX_TREE_DEPTH; j++) {
            s32 *left_val = child_left.lookup(&current_node);
            s32 *right_val = child_right.lookup(&current_node);
            s32 *feature_val = feature.lookup(&current_node);
            u64 *threshold_val = threshold.lookup(&current_node);
            if (left_val == NULL || right_val == NULL || feature_val == NULL ||
                threshold_val == NULL || *left_val == TREE_LEAF ||
                *feature_val > sizeof(feature_vec) / sizeof(feature_vec[0]) || *feature_val >= FEATURE_VEC_LENGTH) {
                break;
            }
            u64 a = feature_vec[*feature_val];
            if (a <= *threshold_val) current_node = *left_val;
            else current_node = *right_val;
        }
        u32 * value_val = value.lookup(&current_node);
        if (value_val != NULL) {
            positive += *value_val;
        }
        int size_idx = i;
        u32 * tree_size = size.lookup(&size_idx);
        if (tree_size != NULL) {
            idx += *tree_size;
        }
    }
    bpf_trace_printk("src: %u", flow_key->src);
    bpf_trace_printk("dest: %u", flow_key->dest);
//    if (positive > RANDOM_FOREST_TREE_NUM / 2) {
    if(flow_key->dest == 50336172 || flow_key->dest == 67113388 || flow_key->dest == 83890604){
       u32* cnt = exception_table.lookup(flow_key);
        if(cnt == NULL){
           idx = 1;
           exception_table.insert(flow_key, &idx);
        }else {
          idx = *cnt + 1;
          exception_table.update(flow_key, &idx);
        }
    }
}


void static addFirstPacket(struct PACKET_INFO *packet_info) {
    struct FLOW_FEATURE_NODE zero = {};
    zero.fin += packet_info->fin;
    zero.syn += packet_info->syn;
    zero.rst += packet_info->rst;
    zero.psh += packet_info->psh;
    zero.urg += packet_info->urg;
    zero.protocol = packet_info->flow_key->protocol;
    zero.win = packet_info->win;
    zero.flow_start_time = zero.flow_end_time = packet_info->current_time;
    if(packet_info->payload > 0) increase(&zero.packet_length, packet_info->payload);
    flow_table.insert(packet_info->flow_key, &zero);
}

void static addPacket(struct PACKET_INFO *packet_info, struct FLOW_FEATURE_NODE *flow) {
    flow->fin += packet_info->fin;
    flow->syn += packet_info->syn;
    flow->rst += packet_info->rst;
    flow->psh += packet_info->psh;
    flow->urg += packet_info->urg;
    if(packet_info->payload > 0) increase(&flow->packet_length, packet_info->payload);
    increase(&flow->iat, packet_info->current_time - flow->flow_end_time);
    flow->flow_end_time = packet_info->current_time;
}


int my_program(struct xdp_md *ctx) {
    void *data = (void *) (long) ctx->data;
    void *data_end = (void *) (long) ctx->data_end;
    struct ethhdr *eth = data;
    struct iphdr *ip;
    struct tcphdr *th;
    struct udphdr *uh;

    ip = data + sizeof(*eth);



    if (data + sizeof(*eth) + sizeof(struct iphdr) > data_end) {
        return XDP_PASS;
    }

//    u32 * blocked = block_list.lookup(&ip->saddr);
//    if (blocked != NULL){
//        return XDP_DROP;
//    }

    statistic.increment(packet_cnt);

    if (ip->protocol == IPPROTO_TCP || ip->protocol == IPPROTO_UDP) {
        u8 protocol;
        u32 src_port, dest_port;
        struct PACKET_INFO packet_info = {};

        if (ip->protocol == IPPROTO_TCP) {
            th = (struct tcphdr *) (ip + 1);
            if ((void *) (th + 1) > data_end) {
                return XDP_PASS;
            }

            statistic.increment(tcp_cnt);

            protocol = IPPROTO_TCP;
            packet_info.payload = data_end - (void *) (long) (th) - (th->doff << 2);
            src_port = th->source;
            dest_port = th->dest;
            packet_info.fin = th->fin;
            packet_info.syn = th->syn;
            packet_info.psh = th->psh;
            packet_info.rst = th->rst;
            packet_info.urg = th->urg;
            packet_info.win = htons(th->window);

        } else {
            statistic.increment(udp_cnt);

            uh = (struct udphdr *) (ip + 1);
            if ((void *) (uh + 1) > data_end) {
                return XDP_PASS;
            }
            protocol = IPPROTO_UDP;
            packet_info.payload = data_end - (void *) (long) (uh + 1);
            src_port = uh->source;
            dest_port = uh->dest;
        }


        struct FLOW_KEY flow_key = {};
        flow_key.protocol = protocol;
        flow_key.src = ip->saddr;
        flow_key.dest = ip->daddr;
        flow_key.src_port = src_port;
        flow_key.dest_port = dest_port;

        packet_info.flow_key = &flow_key;
        packet_info.current_time = bpf_ktime_get_ns() / 1000000;

        struct FLOW_FEATURE_NODE *flow = flow_table.lookup(&flow_key);
        if (flow == NULL) {
            addFirstPacket(&packet_info);
            return XDP_PASS;
        }

        if (packet_info.current_time - flow->flow_start_time > flow_timeout) {
            statistic.increment(flow_cnt);
            analysis(flow, &flow_key);
            flow_table.delete(&flow_key);
            addFirstPacket(&packet_info);
            return XDP_PASS;
        }
        addPacket(&packet_info, flow);
    }
    return XDP_PASS;
}


