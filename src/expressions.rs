use std::hash::Hash;

use ahash::AHashSet;
use petgraph::{algo::tarjan_scc, Directed, Graph};
use pyo3_polars::derive::polars_expr;
use pyo3_polars::export::polars_core::{
    datatypes::DataType, error::PolarsResult, prelude::*, series::Series,
};

trait IntoTypedVec {
    type Output: Clone + Hash + Eq;
    fn to_typed_vec(&self) -> PolarsResult<Vec<Self::Output>>;
    fn vec_to_series(vec: Vec<Self::Output>) -> Series;
}

impl IntoTypedVec for &ChunkedArray<Int64Type> {
    type Output = i64;
    fn to_typed_vec(&self) -> PolarsResult<Vec<i64>> {
        Ok(self.into_iter().flatten().collect())
    }
    fn vec_to_series(vec: Vec<i64>) -> Series {
        Int64Chunked::from_vec("".into(), vec).into_series()
    }
}

impl IntoTypedVec for &ChunkedArray<Int32Type> {
    type Output = i32;
    fn to_typed_vec(&self) -> PolarsResult<Vec<i32>> {
        Ok(self.into_iter().flatten().collect())
    }
    fn vec_to_series(vec: Vec<i32>) -> Series {
        Int32Chunked::from_vec("".into(), vec).into_series()
    }
}

impl IntoTypedVec for &ChunkedArray<StringType> {
    type Output = String;
    fn to_typed_vec(&self) -> PolarsResult<Vec<String>> {
        Ok(self.into_iter().flatten().map(|s| s.to_string()).collect())
    }
    fn vec_to_series(vec: Vec<String>) -> Series {
        let ca: ChunkedArray<StringType> = vec.into_iter().collect();
        ca.into_series()
    }
}

pub fn build_graph<T>(sources: &[T], destinations: &[T]) -> Graph<T, (), Directed>
where
    T: Clone + Hash + Eq,
{
    let mut graph = Graph::new();
    let mut nodes = AHashSet::new();

    // add in all the nodes
    for value in sources.iter().chain(destinations.iter()) {
        if !nodes.contains(value) {
            let graph_value = value.clone();
            nodes.insert(graph_value.clone());
            graph.add_node(graph_value);
        }
    }

    // add all edges
    for (source, dest) in sources.iter().zip(destinations.iter()) {
        let source_index = graph.node_indices().find(|i| &graph[*i] == source).unwrap();

        let dest_index = graph.node_indices().find(|i| &graph[*i] == dest).unwrap();

        graph.add_edge(source_index, dest_index, ());
    }

    graph
}

pub fn get_cycles<T: Clone>(graph: &Graph<T, (), Directed>) -> Vec<Vec<T>> {
    let sccs = tarjan_scc(graph);

    sccs.into_iter()
        .filter(|scc| scc.len() > 1)
        .map(|scc| {
            scc.into_iter()
                .map(|node_index| graph[node_index].clone())
                .collect()
        })
        .collect()
}

pub fn list_dtype(_input_fields: &[Field]) -> PolarsResult<Field> {
    let inner_type = match _input_fields[0].dtype() {
        DataType::Int32 => DataType::Int32,
        DataType::Int64 => DataType::Int64,
        DataType::String => DataType::String,
        dt => {
            return Err(PolarsError::ComputeError(
                format!("Unsupported input type: {}", dt).into(),
            ))
        }
    };
    Ok(Field::new(
        "CyclesFound".into(),
        DataType::List(Box::new(inner_type)),
    ))
}

fn process_cycles<T, C>(sources: &C, destinations: &C) -> PolarsResult<Vec<Series>>
where
    T: Clone + Hash + Eq,
    C: IntoTypedVec<Output = T>,
{
    let sources_vec = sources.to_typed_vec()?;
    let destinations_vec = destinations.to_typed_vec()?;
    let graph = build_graph(&sources_vec, &destinations_vec);
    let cycles = get_cycles(&graph);
    Ok(cycles
        .into_iter()
        .map(|cycle| <C as IntoTypedVec>::vec_to_series(cycle))
        .collect())
}

#[polars_expr(output_type_func=list_dtype)]
pub fn has_cycles(inputs: &[Series]) -> PolarsResult<Series> {
    if inputs[0].dtype() != inputs[1].dtype() {
        return Err(PolarsError::ComputeError(
            "Input columns must have the same datatype".into(),
        ));
    }
    match inputs[0].dtype() {
        DataType::Int64 => {
            let sources = inputs[0].i64()?;
            let destinations = inputs[1].i64()?;
            let list_values = process_cycles(&sources, &destinations)?;
            Ok(Series::new("CyclesFound".into(), list_values))
        }
        DataType::Int32 => {
            let sources = inputs[0].i32()?;
            let destinations = inputs[1].i32()?;
            let list_values = process_cycles(&sources, &destinations)?;
            Ok(Series::new("CyclesFound".into(), list_values))
        }
        DataType::String => {
            let sources = inputs[0].str()?;
            let destinations = inputs[1].str()?;
            let list_values = process_cycles(&sources, &destinations)?;
            Ok(Series::new("CyclesFound".into(), list_values))
        }
        dt => Err(PolarsError::ComputeError(
            format!("Unsupported input type: {}", dt).into(),
        )),
    }
}
