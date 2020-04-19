// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/logging/logging.h"
#include "core/graph/op.h"
#include "core/optimizer/rewrite_rule.h"
#include "orttraining/core/optimizer/memory_swap_rewriter.h"
#include "core/graph/graph_utils.h"

namespace onnxruntime {

static bool IsBackwardNode(const Node& node) {
  return node.Description() == "Backward pass";
}

static void ComputeTopoIndices(const Graph& graph, std::unordered_map<NodeIndex, int>& topo_indices) {
  topo_indices.clear();
  GraphViewer graph_viewer(graph);
  int topo_index = 0;
  for (const auto index : graph_viewer.GetNodesInTopologicalOrder()) {
    topo_indices.insert(std::make_pair(index, topo_index++));
  }
}

Status MemorySwapRewriter::Apply(Graph& graph, Node& src_node, RewriteRuleEffect& rule_effect, const logging::Logger& /*logger*/) const {
  int src_node_output_idx = 0;
  for (auto output_def : src_node.OutputDefs()) {
    NodeArg* src_node_output_arg = const_cast<NodeArg*>(output_def);
    auto& swap_out_arg = graph.GetOrCreateNodeArg(src_node_output_arg->Name() + "_memswap_out", src_node_output_arg->TypeAsProto());
    auto& swap_in_arg = graph.GetOrCreateNodeArg(src_node_output_arg->Name() + "_memswap_in", src_node_output_arg->TypeAsProto());
    auto& swap_out_node = graph.AddNode(src_node_output_arg->Name() + "_swapout",
                                        "SwapToCPU",
                                        "",
                                        {src_node_output_arg},
                                        {&swap_out_arg},
                                        {},
                                        kMSDomain);
    auto& swap_in_node = graph.AddNode(src_node_output_arg->Name() + "_swapin",
                                       "SwapToCPU",
                                       "Backward pass",
                                       {&swap_out_arg},
                                       {&swap_in_arg},
                                       {},
                                       kMSDomain);

    // process output edges from this output_def
    // note this needs to happen before linking src_node with swap_out_node
    const Node* dst_node = nullptr;
    do {
      dst_node = nullptr;
      int dst_arg_idx = -1;
      // note: this loop needs to separate from editing that affects OutputEdges container
      for (auto iter = src_node.OutputEdgesBegin(); iter != src_node.OutputEdgesEnd(); ++iter) {
        if (iter->GetSrcArgIndex() != src_node_output_idx)
          continue;

        if (IsBackwardNode(iter->GetNode())) {
          dst_node = &iter->GetNode();
          dst_arg_idx = iter->GetDstArgIndex();
          break;
        }
      }

      if (dst_node) {
        // remove edge from src_node to dst_node
        graph.RemoveEdge(src_node.Index(), dst_node->Index(), src_node_output_idx, dst_arg_idx);
        // add edge from swap_in to dst_node
        graph.AddEdge(swap_in_node.Index(), dst_node->Index(), 0, dst_arg_idx);
      }
    } while (dst_node != nullptr);

    // add edges in graph
    graph.AddEdge(src_node.Index(), swap_out_node.Index(), src_node_output_idx, 0);
    graph.AddEdge(swap_out_node.Index(), swap_in_node.Index(), 0, 0);

    ++src_node_output_idx;
  }
  rule_effect = RewriteRuleEffect::kModifiedRestOfGraph;
  return Status::OK();
}

bool MemorySwapRewriter::SatisfyCondition(const Graph& graph, const Node& node, const logging::Logger& /*logger*/) const {
  // only check forward nodes
  if (IsBackwardNode(node))
    return false;

  static const Graph* last_graph = nullptr;
  static std::unordered_map<NodeIndex, int> topo_indices;
  if (last_graph != &graph) {
    last_graph = &graph;
    ComputeTopoIndices(graph, topo_indices);
  }

  // check if the node has one output going to a backward
  int fw_topo_idx = topo_indices[node.Index()];
  for (auto iter = node.OutputEdgesBegin(); iter != node.OutputEdgesEnd(); ++iter) {
    if (IsBackwardNode(iter->GetNode())) {
      int bw_topo_idx = topo_indices[iter->GetNode().Index()];
      if (bw_topo_idx - fw_topo_idx > min_topo_distance_)
        return true;
    }
  }
  return false;
}

bool AddControlEdgeForMemorySwapRewriter::SatisfyCondition(const Graph& /*graph*/, const Node& node, const logging::Logger& /*logger*/) const {
  // only add control edge for forward graph
  // backward is OK because topological sort uses reverse DFS
  return !IsBackwardNode(node);
}

Status AddControlEdgeForMemorySwapRewriter::Apply(Graph& graph, Node& node, RewriteRuleEffect& rule_effect, const logging::Logger& /*logger*/) const {
  static const Graph* last_graph = nullptr;
  static std::unordered_map<NodeIndex, int> topo_indices;
  if (last_graph != &graph) {
    last_graph = &graph;
    ComputeTopoIndices(graph, topo_indices);
  }

  // SwapToCPU in forward, need to make sure it happens as early as possible
  int min_topo_index = INT_MAX;
  NodeIndex peer_node_index = 0;
  for (auto in_iter = node.InputEdgesBegin(); in_iter != node.InputEdgesEnd(); ++in_iter) {
    const Node& src_node = in_iter->GetNode();
    for (auto iter = src_node.OutputEdgesBegin(); iter != src_node.OutputEdgesEnd(); ++iter) {
      const auto& peer_node = iter->GetNode();
      if (peer_node.OpType() == "SwapToCPU" || IsBackwardNode(peer_node))
        continue;

      int topo_index = topo_indices[peer_node.Index()];
      if (topo_index < min_topo_index) {
        min_topo_index = topo_index;
        peer_node_index = peer_node.Index();
      }
    }
  }
  if (min_topo_index < INT_MAX) {
    graph.AddControlEdge(node.Index(), peer_node_index);
    rule_effect = RewriteRuleEffect::kModifiedRestOfGraph;
  }
  return Status::OK();
}

}  // namespace onnxruntime
