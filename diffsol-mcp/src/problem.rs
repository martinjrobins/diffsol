use std::{collections::HashMap, sync::Arc};

use diffsol_c::{
    host_array::ToHostArray, LinearSolverType, OdeSolverType, OdeWrapper, SolutionWrapper,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use uuid::Uuid;

use crate::config::ProblemConfig;
use crate::error::DiffsolMcpError;

pub const PROBLEMS_URI: &str = "diffsol://problems";
const PROBLEM_URI_PREFIX: &str = "diffsol://problem/";

pub struct ProblemRecord {
    pub id: Uuid,
    pub name: Option<String>,
    pub code: String,
    pub ode: OdeWrapper,
}

impl ProblemRecord {
    pub fn display_name(&self) -> String {
        self.name
            .clone()
            .unwrap_or_else(|| format!("problem-{}", self.id))
    }

    pub fn resource_uri(&self) -> String {
        format!("{PROBLEM_URI_PREFIX}{}", self.id)
    }

    pub fn summary(&self) -> ProblemSummary {
        ProblemSummary {
            id: self.id.to_string(),
            name: self.name.clone(),
            uri: self.resource_uri(),
        }
    }

    pub fn document(&self) -> Result<ProblemDocument, DiffsolMcpError> {
        Ok(ProblemDocument {
            id: self.id.to_string(),
            name: self.name.clone(),
            uri: self.resource_uri(),
            code: self.code.clone(),
            effective_config: ProblemConfig::from_ode(&self.ode)?,
        })
    }

    pub fn solve(
        &self,
        ode_solver: OdeSolverType,
        linear_solver: LinearSolverType,
        params: &[f64],
        final_time: Option<f64>,
        t_eval: Option<&[f64]>,
    ) -> Result<SolutionWrapper, DiffsolMcpError> {
        self.ode.set_ode_solver(ode_solver)?;
        self.ode.set_linear_solver(linear_solver)?;

        if let Some(t_eval) = t_eval {
            if t_eval.is_empty() {
                return Err(DiffsolMcpError::EmptyTEval);
            }
            return self
                .ode
                .solve_dense(
                    params.to_vec().to_host_array(),
                    t_eval.to_vec().to_host_array(),
                    None,
                )
                .map_err(Into::into);
        }

        let final_time = final_time.ok_or(DiffsolMcpError::MissingSolveTarget)?;
        self.ode
            .solve(params.to_vec().to_host_array(), final_time, None)
            .map_err(Into::into)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
pub struct ProblemSummary {
    pub id: String,
    pub name: Option<String>,
    pub uri: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
pub struct ProblemDocument {
    pub id: String,
    pub name: Option<String>,
    pub uri: String,
    pub code: String,
    pub effective_config: ProblemConfig,
}

#[derive(Default)]
pub struct ProblemStore {
    problems: RwLock<HashMap<Uuid, Arc<ProblemRecord>>>,
}

impl ProblemStore {
    pub async fn insert(&self, record: ProblemRecord) -> Arc<ProblemRecord> {
        let id = record.id;
        let record = Arc::new(record);
        self.problems.write().await.insert(id, record.clone());
        record
    }

    pub async fn list(&self) -> Vec<Arc<ProblemRecord>> {
        let mut records: Vec<_> = self.problems.read().await.values().cloned().collect();
        records.sort_by_key(|record| record.id);
        records
    }

    pub async fn get(&self, id: Uuid) -> Option<Arc<ProblemRecord>> {
        self.problems.read().await.get(&id).cloned()
    }
}

pub fn parse_problem_uri(uri: &str) -> Result<Uuid, DiffsolMcpError> {
    if let Some(value) = uri.strip_prefix(PROBLEM_URI_PREFIX) {
        return Uuid::parse_str(value)
            .map_err(|_| DiffsolMcpError::InvalidProblemUri(uri.to_string()));
    }
    Err(DiffsolMcpError::UnknownResource(uri.to_string()))
}
