use itertools::izip;
use ordered_float::OrderedFloat;
use petgraph::graph::NodeIndex;
use petgraph::stable_graph::StableGraph;
use petgraph::Directed;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_polars::export::polars_core::{datatypes::DataType, prelude::*, series::Series};
use pyo3_polars::PyDataFrame;
use std::collections::HashMap;
use std::hash::Hash;
use thiserror::Error;

#[derive(Clone, Debug, Error)]
enum GraphError {
    #[error("Database connection failed: {0}")]
    MissingData(String),
    #[error("Polars error:{0}")]
    PolarsError(#[from] PolarsError),
    #[error("Invalid Data Format:{0}")]
    InvalidDataType(String),
}
impl From<GraphError> for PyErr {
    fn from(err: GraphError) -> PyErr {
        PyValueError::new_err(err.to_string())
    }
}
#[derive(Clone, Hash, Eq, PartialEq)]
pub enum NodeData {
    Int(i64),
    Float(OrderedFloat<f64>),
    String(String),
}

fn extract_node_data(series: &Series) -> Result<Vec<NodeData>, GraphError> {
    Ok(match series.dtype() {
        DataType::Int64 => series
            .i64()?
            .into_iter()
            .flatten()
            .map(NodeData::Int)
            .collect(),
        DataType::Float64 => series
            .f64()?
            .into_iter()
            .flatten()
            .map(|f| NodeData::Float(OrderedFloat(f)))
            .collect(),
        DataType::String => series
            .str()?
            .into_iter()
            .flatten()
            .map(|s| NodeData::String(s.to_string()))
            .collect(),
        dt => {
            return Err(GraphError::InvalidDataType(format!(
                "Unsupported dtype:{}",
                dt
            )))
        }
    })
}

#[pyclass]
pub struct DirectedGraph {
    graph: StableGraph<NodeData, f64, Directed>,
    hash_map: HashMap<NodeData, NodeIndex>,
}
#[pymethods]
impl DirectedGraph {
    #[new]
    pub fn new(
        pydataframe: PyDataFrame,
        sources_column: String,
        sinks_column: String,
        weights_column: Option<String>,
    ) -> PyResult<Self> {
        let dataframe = pydataframe.clone().0;

        let source = dataframe
            .column(&sources_column)
            .map_err(GraphError::from)?;
        let sink = dataframe.column(&sinks_column).map_err(GraphError::from)?;

        if source.len() != sink.len() {
            return Err(GraphError::MissingData("Missing Datapoints".into()).into());
        }

        let edge_weights;
        let weights: &Series = match &weights_column {
            Some(w) => dataframe
                .column(w)
                .map_err(GraphError::from)?
                .as_materialized_series(),
            None => {
                edge_weights = Series::new("weights".into(), vec![1.0f64; source.len()]);
                &edge_weights
            }
        };

        let weight_vec: Vec<f64> = weights
            .f64()
            .map_err(GraphError::from)?
            .into_iter()
            .flatten()
            .collect();

        let mut graph = StableGraph::<NodeData, f64, Directed>::new();
        let mut node_map = HashMap::new();

        let source_nodes = extract_node_data(source.as_materialized_series())?;
        let sink_nodes = extract_node_data(sink.as_materialized_series())?;

        for node_value in source_nodes.iter().chain(sink_nodes.iter()) {
            if !node_map.contains_key(node_value) {
                let node_index = graph.add_node(node_value.clone());
                node_map.insert(node_value.clone(), node_index);
            }
        }

        for (src, dest, wght) in izip!(source_nodes.iter(), sink_nodes.iter(), weight_vec.iter()) {
            let src_index = node_map[src];
            let dest_index = node_map[dest];
            graph.add_edge(src_index, dest_index, *wght);
        }

        Ok(DirectedGraph {
            graph,
            hash_map: node_map,
        })
    }
}
