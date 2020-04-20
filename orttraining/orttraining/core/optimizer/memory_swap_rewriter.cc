// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/logging/logging.h"
#include "core/graph/op.h"
#include "core/optimizer/rewrite_rule.h"
#include "core/graph/graph_utils.h"
#include "memory_swap_rewriter.h"

namespace onnxruntime {

static bool IsBackwardNode(const Node& node) {
  return node.Description() == "Backward pass";
}

static void ComputeTopoIndices(const Graph& graph, std::unordered_map<NodeIndex, int>& topo_indices) {
  GraphViewer graph_viewer(graph);
  int topo_index = 0;
  topo_indices.clear();
  for (const auto index : graph_viewer.GetNodesInTopologicalOrder()) {
    topo_indices.insert(std::make_pair(index, topo_index++));
  }
}

// we don't want to check these ops for memory swap
static const std::unordered_set<std::string> ignored_op_types =
    {"Shape",
     "Reshape",
     "ReshapeGrad",
     "Transpose"};

Status MemorySwapRewriter::Apply(Graph& graph, Node& src_node, RewriteRuleEffect& rule_effect, const logging::Logger& /*logger*/) const {
  std::unordered_set<int> to_bw_arg_idx;
  for (auto edge_iter = src_node.OutputEdgesBegin(); edge_iter != src_node.OutputEdgesEnd(); ++edge_iter) {
    if (IsBackwardNode(edge_iter->GetNode()) && ignored_op_types.count(edge_iter->GetNode().OpType()) == 0) {
      to_bw_arg_idx.insert(edge_iter->GetSrcArgIndex());
    }
  }
  for (int src_node_output_idx : to_bw_arg_idx) {
    NodeArg* src_node_output_arg = const_cast<NodeArg*>(src_node.OutputDefs()[src_node_output_idx]);
    auto& swap_out_arg = graph.GetOrCreateNodeArg(src_node_output_arg->Name() + "_memswap_out", src_node_output_arg->TypeAsProto());
    auto& swap_in_arg = graph.GetOrCreateNodeArg(src_node_output_arg->Name() + "_memswap_in", src_node_output_arg->TypeAsProto());
    auto& swap_out_node = graph.AddNode(src_node_output_arg->Name() + "_swapout",
                                        "SwapToHost",
                                        "",
                                        {src_node_output_arg},
                                        {&swap_out_arg},
                                        {},
                                        kMSDomain);
    auto& swap_in_node = graph.AddNode(src_node_output_arg->Name() + "_swapin",
                                       "SwapFromHost",
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
    if (IsBackwardNode(iter->GetNode()) && ignored_op_types.count(iter->GetNode().OpType()) == 0) {
      int bw_topo_idx = topo_indices[iter->GetNode().Index()];
      if (bw_topo_idx - fw_topo_idx > min_topo_distance_)
        return true;
    }
  }
  return false;
}

Status AddControlEdgeForMemorySwapRewriter::Apply(Graph& graph, Node& node, RewriteRuleEffect& rule_effect, const logging::Logger& /*logger*/) const {
  static const Graph* last_graph = nullptr;
  static std::unordered_map<NodeIndex, int> topo_indices;
  if (last_graph != &graph) {
    last_graph = &graph;
    ComputeTopoIndices(graph, topo_indices);
  }

  // SwapToHost is in forward, need to make sure it happens as early as possible
  // find the input node (src_node) to SwapToHost, and then find its output node taking the same input as SwapToHost
  // sometimes there might be no node taking the same input as SwapToHost, e.g. saved_mean in LayerNorm
  // in that case, we just find any output of the src_node to SwapToHost
  ORT_ENFORCE(node.GetInputEdgesCount() == 1);
  const auto& src_edge = *(node.InputEdgesBegin());
  const auto& src_node = src_edge.GetNode();
  const auto& src_arg_idx = src_edge.GetSrcArgIndex();

  NodeIndex node_idx = node.Index();
  int min_topo_index = INT_MAX;
  NodeIndex node_found = 0;
  int min_arg_topo_index = INT_MAX;
  NodeIndex arg_node_found = 0;
  for (auto iter = src_node.OutputEdgesBegin(); iter != src_node.OutputEdgesEnd(); ++iter) {
    const Node& peer_node = iter->GetNode();
    if (peer_node.OpType() == "SwapToHost")
      continue;

    int topo_index = topo_indices[peer_node.Index()];
    if (iter->GetSrcArgIndex() == src_arg_idx) {
      if (topo_index < min_topo_index) {
        min_topo_index = topo_index;
        node_found = peer_node.Index();
      }
    } else if (!IsBackwardNode(iter->GetNode())) {
      if (topo_index < min_arg_topo_index) {
        min_arg_topo_index = topo_index;
        arg_node_found = peer_node.Index();
      }
    }
  }
  // add new edge to enforce swap node order, and update precedences
  if (min_topo_index < INT_MAX) {
    graph.AddControlEdge(node_idx, node_found);
  } else if (min_arg_topo_index < INT_MAX) {
    graph.AddControlEdge(node_idx, arg_node_found);
  } else {
    // there could be some optimizations making src_node no longer needed in FW
    // so remove swap nodes
    const auto& swap_in = *node.OutputNodesBegin();
    Node::EdgeSet swap_in_output_edges(swap_in.OutputEdgesBegin(), swap_in.OutputEdgesEnd());
    for (auto edge : swap_in_output_edges) {
      graph.RemoveEdge(swap_in.Index(), edge.GetNode().Index(), edge.GetSrcArgIndex(), edge.GetDstArgIndex());
      graph.AddEdge(src_node.Index(), edge.GetNode().Index(), src_arg_idx, edge.GetDstArgIndex());
    }
    graph.RemoveNode(swap_in.Index());
    graph.RemoveNode(node_idx);
  }
  rule_effect = RewriteRuleEffect::kModifiedRestOfGraph;
  return Status::OK();
}

}  // namespace onnxruntime
