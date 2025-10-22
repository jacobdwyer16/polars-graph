#[cfg(test)]
mod tests {
    use ordered_float::OrderedFloat;
    use polars::prelude::*;
    use polars_graph::graph::{extract_node_data, NodeData};

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
        fn filters_nulls_in_strings() {
            let series = Series::new("nodes".into(), &[Some("A"), None, Some("C")]);
            let result = extract_node_data(&series).unwrap();
            assert_eq!(result.len(), 2);
            assert_eq!(result[0], NodeData::String("A".to_string()));
            assert_eq!(result[1], NodeData::String("C".to_string()));
        }

        #[test]
        fn filters_nulls_in_ints() {
            let series = Series::new("nodes".into(), &[Some(1i64), None, Some(3)]);
            let result = extract_node_data(&series).unwrap();
            assert_eq!(result.len(), 2);
            assert_eq!(result[0], NodeData::Int(1));
            assert_eq!(result[1], NodeData::Int(3));
        }

        #[test]
        fn filters_nulls_in_floats() {
            let series = Series::new("nodes".into(), &[Some(1.5f64), None, Some(3.5)]);
            let result = extract_node_data(&series).unwrap();
            assert_eq!(result.len(), 2);
            assert_eq!(result[0], NodeData::Float(OrderedFloat(1.5)));
            assert_eq!(result[1], NodeData::Float(OrderedFloat(3.5)));
        }

        #[test]
        fn handles_all_nulls() {
            let series = Series::new("nodes".into(), &[None::<i64>, None, None]);
            let result = extract_node_data(&series).unwrap();
            assert_eq!(result.len(), 0);
        }

        #[test]
        fn handles_large_series() {
            let data: Vec<i64> = (0..10000).collect();
            let series = Series::new("nodes".into(), &data);
            let result = extract_node_data(&series).unwrap();
            assert_eq!(result.len(), 10000);
            assert_eq!(result[0], NodeData::Int(0));
            assert_eq!(result[9999], NodeData::Int(9999));
        }

        #[test]
        fn handles_unicode_strings() {
            let series = Series::new("nodes".into(), &["你好", "مرحبا", "Hello", "🎉"]);
            let result = extract_node_data(&series).unwrap();
            assert_eq!(result.len(), 4);
            assert_eq!(result[0], NodeData::String("你好".to_string()));
            assert_eq!(result[1], NodeData::String("مرحبا".to_string()));
            assert_eq!(result[2], NodeData::String("Hello".to_string()));
            assert_eq!(result[3], NodeData::String("🎉".to_string()));
        }

        #[test]
        fn handles_empty_strings() {
            let series = Series::new("nodes".into(), &["", "a", "", "b"]);
            let result = extract_node_data(&series).unwrap();
            assert_eq!(result.len(), 4);
            assert_eq!(result[0], NodeData::String("".to_string()));
            assert_eq!(result[2], NodeData::String("".to_string()));
        }

        #[test]
        fn handles_very_long_strings() {
            let long_str = "a".repeat(10000);
            let series = Series::new("nodes".into(), &[long_str.as_str(), "b"]);
            let result = extract_node_data(&series).unwrap();
            assert_eq!(result.len(), 2);
            assert_eq!(result[0], NodeData::String(long_str));
            assert_eq!(result[1], NodeData::String("b".to_string()));
        }

        #[test]
        fn handles_special_float_values() {
            let series = Series::new(
                "nodes".into(),
                &[f64::INFINITY, f64::NEG_INFINITY, 0.0, -0.0],
            );
            let result = extract_node_data(&series).unwrap();
            assert_eq!(result.len(), 4);
            assert_eq!(result[0], NodeData::Float(OrderedFloat(f64::INFINITY)));
            assert_eq!(result[1], NodeData::Float(OrderedFloat(f64::NEG_INFINITY)));
        }

        #[test]
        fn handles_nan() {
            let series = Series::new("nodes".into(), &[1.0f64, f64::NAN, 2.0]);
            let result = extract_node_data(&series).unwrap();
            assert_eq!(result.len(), 3);
            match &result[1] {
                NodeData::Float(f) => assert!(f.is_nan()),
                _ => panic!("Expected float"),
            }
        }

        #[test]
        fn handles_extreme_integers() {
            let series = Series::new("nodes".into(), &[i64::MIN, 0i64, i64::MAX]);
            let result = extract_node_data(&series).unwrap();
            assert_eq!(result.len(), 3);
            assert_eq!(result[0], NodeData::Int(i64::MIN));
            assert_eq!(result[1], NodeData::Int(0));
            assert_eq!(result[2], NodeData::Int(i64::MAX));
        }

        #[test]
        fn handles_duplicate_values() {
            let series = Series::new("nodes".into(), &[1i64, 1, 2, 2, 3, 3, 3]);
            let result = extract_node_data(&series).unwrap();
            assert_eq!(result.len(), 7);
        }
    }

    mod node_data_behavior {
        use super::*;
        use std::collections::HashSet;

        #[test]
        fn node_data_equality() {
            assert_eq!(NodeData::Int(1), NodeData::Int(1));
            assert_ne!(NodeData::Int(1), NodeData::Int(2));

            assert_eq!(
                NodeData::String("test".to_string()),
                NodeData::String("test".to_string())
            );
            assert_ne!(
                NodeData::String("a".to_string()),
                NodeData::String("b".to_string())
            );

            assert_eq!(
                NodeData::Float(OrderedFloat(1.5)),
                NodeData::Float(OrderedFloat(1.5))
            );
            assert_ne!(
                NodeData::Float(OrderedFloat(1.5)),
                NodeData::Float(OrderedFloat(2.5))
            );
        }

        #[test]
        fn node_data_different_types_not_equal() {
            assert_ne!(NodeData::Int(1), NodeData::Float(OrderedFloat(1.0)));
            assert_ne!(NodeData::Int(1), NodeData::String("1".to_string()));
            assert_ne!(
                NodeData::Float(OrderedFloat(1.0)),
                NodeData::String("1.0".to_string())
            );
        }

        #[test]
        fn node_data_hashable() {
            let mut set = HashSet::new();
            set.insert(NodeData::Int(1));
            set.insert(NodeData::Int(2));
            set.insert(NodeData::Int(1));

            assert_eq!(set.len(), 2);
            assert!(set.contains(&NodeData::Int(1)));
            assert!(set.contains(&NodeData::Int(2)));
        }

        #[test]
        fn node_data_clone() {
            let node = NodeData::String("test".to_string());
            let cloned = node.clone();
            assert_eq!(node, cloned);
        }
    }
}
