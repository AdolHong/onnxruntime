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

// we don't want to check these ops for memory swap
static const std::unordered_set<std::string> ignored_op_types =
    {"SwapToCPU",
     "Shape",
     "ConstantOfShape",
     "Expand",
     "Slice",
     "Gather",
     "Concat"};

bool MemorySwapRewriter::SatisfyCondition(const Graph& graph, const Node& node, const logging::Logger& /*logger*/) const {
  if (ignored_op_types.count(node.OpType()))
    return false;

  // only check forward nodes
  if (IsBackwardNode(node))
    return false;

  static const Graph* last_graph = nullptr;
  static std::unordered_map<NodeIndex, int> topo_indices;
  if (last_graph != &graph) {
    last_graph = &graph;
    topo_indices.clear();
    GraphViewer graph_viewer(graph);
    int topo_index = 0;
    for (const auto index : graph_viewer.GetNodesInTopologicalOrder()) {
      topo_indices.insert(std::make_pair(index, topo_index++));
    }
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

Status AddControlEdgeForMemorySwapRewriter::Apply(Graph& graph, Node& node, RewriteRuleEffect& rule_effect, const logging::Logger& /*logger*/) const {
  static const Graph* last_graph = nullptr;
  static std::unordered_map<NodeIndex, std::unordered_set<NodeIndex>> precedences;

  auto update_precedences = []() {
    GraphViewer gv(*last_graph);
    precedences.clear();
    for (auto i : gv.GetNodesInTopologicalOrder()) {
      const Node* pnode = gv.GetNode(i);
      precedences.insert(std::make_pair(i, std::unordered_set<NodeIndex>()));
      for (auto iter = pnode->InputNodesBegin(); iter != pnode->InputNodesEnd(); ++iter) {
        NodeIndex prev_node_idx = iter->Index();
        precedences[i].insert(precedences[prev_node_idx].begin(), precedences[prev_node_idx].end());
        ORT_ENFORCE(precedences[i].count(i) == 0);
      }
    }
  };

  if (last_graph != &graph) {
    last_graph = &graph;
    update_precedences();
  }

  NodeIndex swap_node_idx = node.Index();
  NodeIndex node_found = 0;
  bool found = false;

  // don't build control edges on these ops
  if (IsBackwardNode(node)) {
    // SwapToCPU in backward, need to make sure it happens as late as possible
    std::function<bool(NodeIndex)> find_in_bw =
        [&](NodeIndex node_to_search) {
          std::vector<NodeIndex> prev_nodes_to_search;
          const Node* pnode = graph.GetNode(node_to_search);
          if (pnode == nullptr || ignored_op_types.count(pnode->OpType()) || !IsBackwardNode(*pnode))
            return false;

          for (auto iter = pnode->InputNodesBegin(); iter != pnode->InputNodesEnd(); ++iter) {
            if (ignored_op_types.count(iter->OpType()) || !IsBackwardNode(*iter))
              continue;

            if (precedences[iter->Index()].count(swap_node_idx) == 0) {
              node_found = iter->Index();
              return true;
            }
            prev_nodes_to_search.push_back(iter->Index());
          }
          // recursive if not found yet
          for (auto prev : prev_nodes_to_search) {
            if (find_in_bw(prev))
              return true;
          }
          return false;
        };

    for (auto out_iter = node.OutputEdgesBegin(); out_iter != node.OutputEdgesEnd(); ++out_iter) {
      if (find_in_bw(out_iter->GetNode().Index())) {
        found = true;
        break;
      }
    }
    if (found) {
      // add new edge to enforce swap node order, and update precedences
      graph.AddControlEdge(node_found, swap_node_idx);
      update_precedences();
      rule_effect = RewriteRuleEffect::kModifiedRestOfGraph;
    }
  } else {
    // SwapToCPU in forward, need to make sure it happens as early as possible
    std::function<bool(NodeIndex)> find_in_fw =
        [&](NodeIndex node_to_search) {
          std::vector<NodeIndex> next_nodes_to_search;
          const Node* pnode = graph.GetNode(node_to_search);
          if (pnode == nullptr || ignored_op_types.count(pnode->OpType()) || IsBackwardNode(*pnode))
            return false;

          for (auto iter = pnode->OutputNodesBegin(); iter != pnode->OutputNodesEnd(); ++iter) {
            if (ignored_op_types.count(iter->OpType()) || IsBackwardNode(*iter))
              continue;

            if (precedences[swap_node_idx].count(iter->Index()) == 0) {
              node_found = iter->Index();
              return true;
            }
            next_nodes_to_search.push_back(iter->Index());
          }
          // recursive if not found yet
          for (auto next : next_nodes_to_search) {
            if (find_in_fw(next))
              return true;
          }
          return false;
        };

    for (auto in_iter = node.InputEdgesBegin(); in_iter != node.InputEdgesEnd(); ++in_iter) {
      if (find_in_fw(in_iter->GetNode().Index())) {
        found = true;
        break;
      }
    }
    if (found) {
      // add new edge to enforce swap node order, and update precedences
      graph.AddControlEdge(swap_node_idx, node_found);
      update_precedences();
      rule_effect = RewriteRuleEffect::kModifiedRestOfGraph;
    }
  }
  return Status::OK();
}

}  // namespace onnxruntime
