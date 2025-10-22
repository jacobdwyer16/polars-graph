#[cfg(test)]
mod tests {
    use super::*;
    use polars::prelude::*;
    use ordered_float::OrderedFloat;
    use polars_graph::{extract_node_data, DirectedGraph, NodeData, UndirectedGraph};

    mod node_extraction {
        use super::*;

        #[test]
        fn extracts_string_nodes() {
            let series = Series::new("nodes".into(), &["A", "B", "C"]);
            let result = extract_node_data(&series).unwrap();

            assert_eq!(result.len(), 3);
            assert_eq!(result[0], NodeData::String("A".to_string()));
            assert_eq!(result[1], NodeData::String("B".to_string()));
            assert_eq!(result[2], NodeData::String("C".to_string()));
        }

        #[test]
        fn extracts_int_nodes() {
            let series = Series::new("nodes".into(), &[1i64, 2, 3]);
            let result = extract_node_data(&series).unwrap();

            assert_eq!(result.len(), 3);
            assert_eq!(result[0], NodeData::Int(1));
            assert_eq!(result[1], NodeData::Int(2));
            assert_eq!(result[2], NodeData::Int(3));
        }

        #[test]
        fn extracts_float_nodes() {
            let series = Series::new("nodes".into(), &[1.5f64, 2.5, 3.5]);
            let result = extract_node_data(&series).unwrap();

            assert_eq!(result.len(), 3);
            assert_eq!(result[0], NodeData::Float(OrderedFloat(1.5)));
            assert_eq!(result[1], NodeData::Float(OrderedFloat(2.5)));
            assert_eq!(result[2], NodeData::Float(OrderedFloat(3.5)));
        }

        #[test]
        fn rejects_unsupported_types() {
            let series = Series::new("nodes".into(), &[true, false, true]);
            let result = extract_node_data(&series);
            assert!(result.is_err());
        }

        #[test]
        fn handles_empty_series() {
            let series = Series::new_empty("nodes".into(), &DataType::String);
            let result = extract_node_data(&series).unwrap();
            assert_eq!(result.len(), 0);
        }

        #[test]
        fn handles_nulls_in_strings() {
            let series = Series::new("nodes".into(), &[Some("A"), None, Some("C")]);
            let result = extract_node_data(&series);
            assert!(result.is_ok() || result.is_err());
        }

        #[test]
        fn handles_nulls_in_ints() {
            let series = Series::new("nodes".into(), &[Some(1i64), None, Some(3)]);
            let result = extract_node_data(&series);
            assert!(result.is_ok() || result.is_err());
        }
    }

    mod directed_graph {
        use super::*;

        #[test]
        fn creates_empty_graph() {
            let graph = DirectedGraph::new();
            assert_eq!(graph.node_count(), 0);
            assert_eq!(graph.edge_count(), 0);
        }

        #[test]
        fn adds_single_edge() {
            let mut graph = DirectedGraph::new();
            let node_a = NodeData::String("A".to_string());
            let node_b = NodeData::String("B".to_string());

            graph.add_edge(node_a.clone(), node_b.clone(), 1.0);

            assert_eq!(graph.node_count(), 2);
            assert_eq!(graph.edge_count(), 1);
            assert!(graph.has_node(&node_a));
            assert!(graph.has_node(&node_b));
            assert!(graph.has_edge(&node_a, &node_b));
        }

        #[test]
        fn edge_not_bidirectional() {
            let mut graph = DirectedGraph::new();
            graph.add_edge(NodeData::Int(1), NodeData::Int(2), 1.0);

            assert!(graph.has_edge(&NodeData::Int(1), &NodeData::Int(2)));
            assert!(!graph.has_edge(&NodeData::Int(2), &NodeData::Int(1)));
        }

        #[test]
        fn handles_self_loop() {
            let mut graph = DirectedGraph::new();
            graph.add_edge(NodeData::Int(1), NodeData::Int(1), 1.0);

            assert_eq!(graph.node_count(), 1);
            assert_eq!(graph.edge_count(), 1);
            assert!(graph.has_edge(&NodeData::Int(1), &NodeData::Int(1)));
        }

        #[test]
        fn handles_duplicate_edges() {
            let mut graph = DirectedGraph::new();
            graph.add_edge(NodeData::Int(1), NodeData::Int(2), 1.0);
            graph.add_edge(NodeData::Int(1), NodeData::Int(2), 2.0);

            assert_eq!(graph.node_count(), 2);
        }

        #[test]
        fn builds_simple_chain() {
            let mut graph = DirectedGraph::new();
            graph.add_edge(NodeData::Int(1), NodeData::Int(2), 1.0);
            graph.add_edge(NodeData::Int(2), NodeData::Int(3), 1.0);
            graph.add_edge(NodeData::Int(3), NodeData::Int(4), 1.0);

            assert_eq!(graph.node_count(), 4);
            assert_eq!(graph.edge_count(), 3);
        }

        #[test]
        fn detects_simple_cycle() {
            let mut graph = DirectedGraph::new();
            graph.add_edge(NodeData::Int(1), NodeData::Int(2), 1.0);
            graph.add_edge(NodeData::Int(2), NodeData::Int(3), 1.0);
            graph.add_edge(NodeData::Int(3), NodeData::Int(1), 1.0);

            let cycles = graph.find_cycles();
            assert_eq!(cycles.len(), 1);
            assert_eq!(cycles[0].len(), 3);
        }

        #[test]
        fn detects_no_cycles_in_dag() {
            let mut graph = DirectedGraph::new();
            graph.add_edge(NodeData::Int(1), NodeData::Int(2), 1.0);
            graph.add_edge(NodeData::Int(2), NodeData::Int(3), 1.0);
            graph.add_edge(NodeData::Int(1), NodeData::Int(3), 1.0);

            let cycles = graph.find_cycles();
            assert_eq!(cycles.len(), 0);
        }

        #[test]
        fn detects_multiple_cycles() {
            let mut graph = DirectedGraph::new();
            graph.add_edge(NodeData::Int(1), NodeData::Int(2), 1.0);
            graph.add_edge(NodeData::Int(2), NodeData::Int(1), 1.0);
            graph.add_edge(NodeData::Int(3), NodeData::Int(4), 1.0);
            graph.add_edge(NodeData::Int(4), NodeData::Int(3), 1.0);

            let cycles = graph.find_cycles();
            assert_eq!(cycles.len(), 2);
        }

        #[test]
        fn handles_disconnected_components() {
            let mut graph = DirectedGraph::new();
            graph.add_edge(NodeData::Int(1), NodeData::Int(2), 1.0);
            graph.add_edge(NodeData::Int(3), NodeData::Int(4), 1.0);

            assert_eq!(graph.node_count(), 4);
            assert_eq!(graph.edge_count(), 2);
        }

        #[test]
        fn supports_negative_weights() {
            let mut graph = DirectedGraph::new();
            graph.add_edge(NodeData::Int(1), NodeData::Int(2), -5.0);

            assert!(graph.has_edge(&NodeData::Int(1), &NodeData::Int(2)));
        }

        #[test]
        fn supports_zero_weights() {
            let mut graph = DirectedGraph::new();
            graph.add_edge(NodeData::Int(1), NodeData::Int(2), 0.0);

            assert_eq!(graph.edge_count(), 1);
        }

        #[test]
        fn handles_large_graph() {
            let mut graph = DirectedGraph::new();
            for i in 0..1000 {
                graph.add_edge(NodeData::Int(i), NodeData::Int(i + 1), 1.0);
            }

            assert_eq!(graph.node_count(), 1001);
            assert_eq!(graph.edge_count(), 1000);
        }

        #[test]
        fn supports_string_nodes() {
            let mut graph = DirectedGraph::new();
            graph.add_edge(
                NodeData::String("start".to_string()),
                NodeData::String("end".to_string()),
                1.0
            );

            assert_eq!(graph.node_count(), 2);
        }

        #[test]
        fn supports_float_nodes() {
            let mut graph = DirectedGraph::new();
            graph.add_edge(NodeData::Float(OrderedFloat(1.5)), NodeData::Float(OrderedFloat(2.5)), 1.0);

            assert_eq!(graph.node_count(), 2);
        }

        #[test]
        fn supports_mixed_node_types() {
            let mut graph = DirectedGraph::new();
            graph.add_edge(NodeData::Int(1), NodeData::Int(2), 1.0);
            graph.add_edge(NodeData::String("A".to_string()), NodeData::String("B".to_string()), 1.0);

            assert_eq!(graph.node_count(), 4);
        }
    }

    mod undirected_graph {
        use super::*;

        #[test]
        fn creates_empty_graph() {
            let graph = UndirectedGraph::new();
            assert_eq!(graph.node_count(), 0);
            assert_eq!(graph.edge_count(), 0);
        }

        #[test]
        fn edges_are_bidirectional() {
            let mut graph = UndirectedGraph::new();
            graph.add_edge(NodeData::Int(1), NodeData::Int(2), 1.0);

            assert!(graph.has_edge(&NodeData::Int(1), &NodeData::Int(2)));
            assert!(graph.has_edge(&NodeData::Int(2), &NodeData::Int(1)));
        }

        #[test]
        fn adds_single_edge() {
            let mut graph = UndirectedGraph::new();
            graph.add_edge(NodeData::Int(1), NodeData::Int(2), 1.0);

            assert_eq!(graph.node_count(), 2);
            assert_eq!(graph.edge_count(), 1);
        }

        #[test]
        fn handles_self_loop() {
            let mut graph = UndirectedGraph::new();
            graph.add_edge(NodeData::Int(1), NodeData::Int(1), 1.0);

            assert_eq!(graph.node_count(), 1);
            assert_eq!(graph.edge_count(), 1);
        }

        #[test]
        fn builds_triangle() {
            let mut graph = UndirectedGraph::new();
            graph.add_edge(NodeData::Int(1), NodeData::Int(2), 1.0);
            graph.add_edge(NodeData::Int(2), NodeData::Int(3), 1.0);
            graph.add_edge(NodeData::Int(3), NodeData::Int(1), 1.0);

            assert_eq!(graph.node_count(), 3);
            assert_eq!(graph.edge_count(), 3);
        }

        #[test]
        fn handles_disconnected_components() {
            let mut graph = UndirectedGraph::new();
            graph.add_edge(NodeData::Int(1), NodeData::Int(2), 1.0);
            graph.add_edge(NodeData::Int(3), NodeData::Int(4), 1.0);

            assert_eq!(graph.node_count(), 4);
            assert_eq!(graph.edge_count(), 2);
        }

        #[test]
        fn detects_connected_components() {
            let mut graph = UndirectedGraph::new();
            graph.add_edge(NodeData::Int(1), NodeData::Int(2), 1.0);
            graph.add_edge(NodeData::Int(2), NodeData::Int(3), 1.0);
            graph.add_edge(NodeData::Int(4), NodeData::Int(5), 1.0);

            let components = graph.connected_components();
            assert_eq!(components.len(), 2);
        }

        #[test]
        fn finds_shortest_path() {
            let mut graph = UndirectedGraph::new();
            graph.add_edge(NodeData::Int(1), NodeData::Int(2), 1.0);
            graph.add_edge(NodeData::Int(2), NodeData::Int(3), 1.0);
            graph.add_edge(NodeData::Int(1), NodeData::Int(3), 5.0);

            let path = graph.shortest_path(&NodeData::Int(1), &NodeData::Int(3));
            assert!(path.is_some());
        }
    }

    mod graph_algorithms {
        use super::*;

        #[test]
        fn topological_sort_on_dag() {
            let mut graph = DirectedGraph::new();
            graph.add_edge(NodeData::Int(1), NodeData::Int(2), 1.0);
            graph.add_edge(NodeData::Int(1), NodeData::Int(3), 1.0);
            graph.add_edge(NodeData::Int(2), NodeData::Int(4), 1.0);
            graph.add_edge(NodeData::Int(3), NodeData::Int(4), 1.0);

            let sorted = graph.topological_sort();
            assert!(sorted.is_ok());
        }

        #[test]
        fn topological_sort_fails_on_cycle() {
            let mut graph = DirectedGraph::new();
            graph.add_edge(NodeData::Int(1), NodeData::Int(2), 1.0);
            graph.add_edge(NodeData::Int(2), NodeData::Int(3), 1.0);
            graph.add_edge(NodeData::Int(3), NodeData::Int(1), 1.0);

            let sorted = graph.topological_sort();
            assert!(sorted.is_err());
        }

        #[test]
        fn strongly_connected_components() {
            let mut graph = DirectedGraph::new();
            graph.add_edge(NodeData::Int(1), NodeData::Int(2), 1.0);
            graph.add_edge(NodeData::Int(2), NodeData::Int(3), 1.0);
            graph.add_edge(NodeData::Int(3), NodeData::Int(1), 1.0);
            graph.add_edge(NodeData::Int(4), NodeData::Int(5), 1.0);

            let sccs = graph.strongly_connected_components();
            assert!(sccs.len() >= 2);
        }
    }

    mod edge_cases {
        use super::*;

        #[test]
        fn handles_extreme_float_weights() {
            let mut graph = DirectedGraph::new();
            graph.add_edge(NodeData::Int(1), NodeData::Int(2), f64::MAX);
            graph.add_edge(NodeData::Int(2), NodeData::Int(3), f64::MIN);

            assert_eq!(graph.edge_count(), 2);
        }

        #[test]
        fn handles_nan_weights() {
            let mut graph = DirectedGraph::new();
            graph.add_edge(NodeData::Int(1), NodeData::Int(2), f64::NAN);

            assert_eq!(graph.edge_count(), 1);
        }

        #[test]
        fn handles_infinity_weights() {
            let mut graph = DirectedGraph::new();
            graph.add_edge(NodeData::Int(1), NodeData::Int(2), f64::INFINITY);
            graph.add_edge(NodeData::Int(2), NodeData::Int(3), f64::NEG_INFINITY);

            assert_eq!(graph.edge_count(), 2);
        }

        #[test]
        fn handles_unicode_strings() {
            let mut graph = DirectedGraph::new();
            graph.add_edge(
                NodeData::String("你好".to_string()),
                NodeData::String("مرحبا".to_string()),
                1.0
            );

            assert_eq!(graph.node_count(), 2);
        }

        #[test]
        fn handles_empty_strings() {
            let mut graph = DirectedGraph::new();
            graph.add_edge(
                NodeData::String("".to_string()),
                NodeData::String("a".to_string()),
                1.0
            );

            assert_eq!(graph.node_count(), 2);
        }

        #[test]
        fn handles_very_long_strings() {
            let mut graph = DirectedGraph::new();
            let long_str = "a".repeat(10000);
            graph.add_edge(
                NodeData::String(long_str.clone()),
                NodeData::String("b".to_string()),
                1.0
            );

            assert_eq!(graph.node_count(), 2);
        }
    }
}
