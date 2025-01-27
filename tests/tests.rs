#[cfg(test)]
mod tests {
    use crate::expressions::*;

    #[test]
    fn test_build_graph() {
        let sources = vec![1, 2, 3];
        let destinations = vec![2, 3, 1];
        let graph = build_graph(&sources, &destinations);

        // Test number of nodes
        assert_eq!(graph.node_count(), 3);
        // Test number of edges
        assert_eq!(graph.edge_count(), 3);
    }

    #[test]
    fn test_get_cycles_int32() {
        let sources = vec![1, 2, 3, 4];
        let destinations = vec![2, 3, 1, 5];
        let graph = build_graph(&sources, &destinations);

        let cycles = get_cycles(&graph);
        assert_eq!(cycles.len(), 1); // Should find one cycle
        assert!(cycles[0].contains(&1)); // Cycle should contain these nodes
        assert!(cycles[0].contains(&2));
        assert!(cycles[0].contains(&3));
    }

    #[test]
    fn test_get_cycles_strings() {
        let sources = vec!["A", "B", "C", "D"];
        let destinations = vec!["B", "C", "A", "E"];
        let graph = build_graph(&sources, &destinations);

        let cycles = get_cycles(&graph);
        assert_eq!(cycles.len(), 1); // Should find one cycle
        assert!(cycles[0].contains(&"A")); // Cycle should contain these nodes
        assert!(cycles[0].contains(&"B"));
        assert!(cycles[0].contains(&"C"));
    }

    #[test]
    fn test_get_cycles_int64() {
        let sources = vec![1i64, 2i64, 3i64, 4i64];
        let destinations = vec![2i64, 3i64, 1i64, 5i64];
        let graph = build_graph(&sources, &destinations);

        let cycles = get_cycles(&graph);
        assert_eq!(cycles.len(), 1); // Should find one cycle
        assert!(cycles[0].contains(&1i64)); // Cycle should contain these nodes
        assert!(cycles[0].contains(&2i64));
        assert!(cycles[0].contains(&3i64));
    }

    #[test]
    fn test_get_cycles_no_cycle() {
        let sources = vec![1i64, 2i64, 3i64, 4i64];
        let destinations = vec![2i64, 3i64, 4i64, 5i64];
        let graph = build_graph(&sources, &destinations);

        let cycles = get_cycles(&graph);
        assert_eq!(cycles.len(), 0); // Should find no cycles
    }
}
