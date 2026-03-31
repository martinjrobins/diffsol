use std::{future::Future, sync::Arc};

use diffsol_c::{LinearSolverType, OdeSolverType, SolutionWrapper};
use rmcp::{
    handler::server::{router::tool::ToolRouter, wrapper::Parameters},
    model::{
        AnnotateAble, CallToolResult, Content, ListResourceTemplatesResult, ListResourcesResult,
        RawResource, RawResourceTemplate, ReadResourceRequestParams, ReadResourceResult,
        ResourceContents, ServerCapabilities, ServerInfo,
    },
    tool, tool_handler, tool_router, Json, RoleServer, ServerHandler, ServiceExt,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::{
    config::{ProblemConfig, ProblemConfigInput},
    error::DiffsolMcpError,
    problem::{parse_problem_uri, ProblemRecord, ProblemStore, ProblemSummary, PROBLEMS_URI},
};

const DIFFSL_DOCS_URI: &str = "diffsol://docs/diffsl";
const DIFFSL_DOCS_TEXT: &str = include_str!("../docs/diffsl.md");

#[derive(Clone)]
pub struct DiffsolMcpServer {
    tool_router: ToolRouter<Self>,
    problems: Arc<ProblemStore>,
}

impl DiffsolMcpServer {
    pub fn new() -> Self {
        Self {
            tool_router: Self::tool_router(),
            problems: Arc::new(ProblemStore::default()),
        }
    }

    async fn list_problem_summaries(&self) -> Vec<ProblemSummary> {
        self.problems
            .list()
            .await
            .into_iter()
            .map(|problem| problem.summary())
            .collect()
    }

    async fn get_problem(&self, id: Uuid) -> Result<Arc<ProblemRecord>, DiffsolMcpError> {
        self.problems
            .get(id)
            .await
            .ok_or(DiffsolMcpError::ProblemNotFound(id))
    }
}

impl Default for DiffsolMcpServer {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
pub struct CreateProblemRequest {
    pub code: String,
    pub config: Option<ProblemConfigInput>,
    pub name: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
pub struct CreateProblemResponse {
    pub id: String,
    pub name: Option<String>,
    pub uri: String,
    pub effective_config: ProblemConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
pub struct SolveProblemRequest {
    pub problem_id: String,
    pub ode_solver: OdeSolverType,
    pub linear_solver: LinearSolverType,
    pub params: Vec<f64>,
    pub final_time: Option<f64>,
    pub t_eval: Option<Vec<f64>>,
}

#[derive(Clone, Serialize)]
pub struct SolveProblemResponse {
    pub problem_id: String,
    pub ode_solver: OdeSolverType,
    pub linear_solver: LinearSolverType,
    #[serde(flatten)]
    pub solution: SolutionWrapper,
}

#[tool_router(router = tool_router)]
impl DiffsolMcpServer {
    #[tool(
        name = "create_problem",
        description = "Create a session-scoped diffsol problem from DiffSL source and optional configuration"
    )]
    pub async fn create_problem(
        &self,
        Parameters(request): Parameters<CreateProblemRequest>,
    ) -> Result<Json<CreateProblemResponse>, String> {
        let ode = ProblemConfig::build_ode(&request.code, request.config.as_ref())
            .map_err(|error| error.to_string())?;
        let effective_config = ProblemConfig::from_ode(&ode).map_err(|error| error.to_string())?;

        let record = ProblemRecord {
            id: Uuid::new_v4(),
            name: request.name.clone(),
            code: request.code,
            ode,
        };

        let record = self.problems.insert(record).await;

        Ok(Json(CreateProblemResponse {
            id: record.id.to_string(),
            name: record.name.clone(),
            uri: record.resource_uri(),
            effective_config,
        }))
    }

    #[tool(
        name = "solve_problem",
        description = "Solve a previously created diffsol problem with a selected ODE solver and linear solver"
    )]
    pub async fn solve_problem(
        &self,
        Parameters(request): Parameters<SolveProblemRequest>,
    ) -> Result<CallToolResult, rmcp::ErrorData> {
        let problem_id = Uuid::parse_str(&request.problem_id)
            .map_err(|error| rmcp::ErrorData::invalid_params(error.to_string(), None))?;
        let record = self.get_problem(problem_id).await.map_err(invalid_params)?;

        let solution = record
            .solve(
                request.ode_solver,
                request.linear_solver,
                &request.params,
                request.final_time,
                request.t_eval.as_deref(),
            )
            .map_err(invalid_params)?;

        let response = SolveProblemResponse {
            problem_id: record.id.to_string(),
            ode_solver: request.ode_solver,
            linear_solver: request.linear_solver,
            solution,
        };

        let content = Content::json(response)
            .map_err(|error| rmcp::ErrorData::internal_error(error.to_string(), None))?;
        Ok(CallToolResult::success(vec![content]))
    }
}

#[tool_handler(router = self.tool_router)]
impl ServerHandler for DiffsolMcpServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo::new(
            ServerCapabilities::builder()
                .enable_tools()
                .enable_resources()
                .build(),
        )
        .with_instructions(
            "Create session-scoped diffsol ODE problems from DiffSL code, inspect them as resources, solve them with runtime-selected solvers, and read `diffsol://docs/diffsl` for the DiffSL syntax reference.",
        )
    }

    fn list_resources(
        &self,
        _request: Option<rmcp::model::PaginatedRequestParams>,
        _context: rmcp::service::RequestContext<RoleServer>,
    ) -> impl Future<Output = Result<ListResourcesResult, rmcp::ErrorData>> + Send + '_ {
        async move {
            let mut resources = vec![
                RawResource::new(DIFFSL_DOCS_URI, "DiffSL DSL reference")
                    .with_title("DiffSL DSL reference")
                    .with_description("Markdown reference for DiffSL syntax, tensors, ODE definitions, and grammar")
                    .with_mime_type("text/markdown")
                    .no_annotation(),
                RawResource::new(PROBLEMS_URI, "diffsol problems")
                    .with_title("diffsol session problems")
                    .with_description("JSON list of all problems created in this MCP session")
                    .with_mime_type("application/json")
                    .no_annotation(),
            ];

            resources.extend(self.problems.list().await.into_iter().map(|problem| {
                RawResource::new(problem.resource_uri(), problem.display_name())
                    .with_description("A session-scoped diffsol problem")
                    .with_mime_type("application/json")
                    .no_annotation()
            }));

            Ok(ListResourcesResult::with_all_items(resources))
        }
    }

    fn list_resource_templates(
        &self,
        _request: Option<rmcp::model::PaginatedRequestParams>,
        _context: rmcp::service::RequestContext<RoleServer>,
    ) -> impl Future<Output = Result<ListResourceTemplatesResult, rmcp::ErrorData>> + Send + '_
    {
        async move {
            Ok(ListResourceTemplatesResult::with_all_items(vec![
                RawResourceTemplate::new("diffsol://problem/{problem_id}", "diffsol problem")
                    .with_description("Read a previously created diffsol problem by id")
                    .with_mime_type("application/json")
                    .no_annotation(),
            ]))
        }
    }

    fn read_resource(
        &self,
        request: ReadResourceRequestParams,
        _context: rmcp::service::RequestContext<RoleServer>,
    ) -> impl Future<Output = Result<ReadResourceResult, rmcp::ErrorData>> + Send + '_ {
        async move {
            if request.uri == DIFFSL_DOCS_URI {
                return Ok(text_resource(
                    DIFFSL_DOCS_URI,
                    DIFFSL_DOCS_TEXT,
                    "text/markdown",
                ));
            }

            if request.uri == PROBLEMS_URI {
                let summaries = self.list_problem_summaries().await;
                return Ok(json_resource(PROBLEMS_URI, &summaries)?);
            }

            let problem_id = parse_problem_uri(&request.uri).map_err(invalid_params)?;
            let document = self
                .get_problem(problem_id)
                .await
                .map_err(invalid_params)?
                .document()
                .map_err(invalid_params)?;
            Ok(json_resource(&request.uri, &document)?)
        }
    }
}

pub async fn run_stdio() -> Result<(), Box<dyn std::error::Error>> {
    let server = DiffsolMcpServer::new()
        .serve(rmcp::transport::stdio())
        .await?;
    server.waiting().await?;
    Ok(())
}

fn json_resource(uri: &str, value: &impl Serialize) -> Result<ReadResourceResult, rmcp::ErrorData> {
    let text = serde_json::to_string_pretty(value)
        .map_err(|error| rmcp::ErrorData::internal_error(error.to_string(), None))?;
    Ok(text_resource(uri, &text, "application/json"))
}

fn text_resource(uri: &str, text: &str, mime_type: &str) -> ReadResourceResult {
    ReadResourceResult::new(vec![
        ResourceContents::text(text.to_string(), uri).with_mime_type(mime_type)
    ])
}

fn invalid_params(error: DiffsolMcpError) -> rmcp::ErrorData {
    rmcp::ErrorData::invalid_params(error.to_string(), None)
}

#[cfg(test)]
mod tests {
    use anyhow::Result;
    use rmcp::{
        model::{CallToolRequestParams, ReadResourceRequestParams, ResourceContents},
        ServiceExt,
    };
    use serde::Deserialize;
    use serde_json::Value;

    use super::*;
    use crate::problem::ProblemDocument;

    const LOGISTIC_DIFFSL: &str = r#"
        r { 2.0 }
        u_i { y = 0.1 }
        F_i { (r * y) * (1 - y) }
    "#;

    fn tool_args<T: Serialize>(value: &T) -> serde_json::Map<String, Value> {
        serde_json::to_value(value)
            .unwrap()
            .as_object()
            .unwrap()
            .clone()
    }

    fn read_text(contents: &[ResourceContents]) -> String {
        match &contents[0] {
            ResourceContents::TextResourceContents { text, .. } => text.clone(),
            ResourceContents::BlobResourceContents { .. } => {
                panic!("expected text resource contents")
            }
        }
    }

    #[derive(Debug, Deserialize)]
    struct SolveProblemResponse {
        problem_id: String,
        ts: Vec<f64>,
        ys: Vec<Vec<f64>>,
        current_state: Vec<f64>,
    }

    #[tokio::test]
    async fn create_problem_exposes_resource_and_solves() -> Result<()> {
        let (server_transport, client_transport) = tokio::io::duplex(16 * 1024);
        tokio::spawn(async move {
            let server = DiffsolMcpServer::new().serve(server_transport).await?;
            server.waiting().await?;
            anyhow::Ok(())
        });

        let client = ().serve(client_transport).await?;

        let create_request = CreateProblemRequest {
            code: LOGISTIC_DIFFSL.to_string(),
            config: Some(ProblemConfigInput {
                rtol: Some(1e-8),
                atol: Some(1e-8),
                ..Default::default()
            }),
            name: Some("logistic".to_string()),
        };

        let created: CreateProblemResponse = client
            .peer()
            .call_tool(
                CallToolRequestParams::new("create_problem")
                    .with_arguments(tool_args(&create_request)),
            )
            .await?
            .into_typed()?;

        let resources = client.peer().list_all_resources().await?;
        assert_eq!(resources.len(), 3);
        assert!(resources
            .iter()
            .any(|resource| resource.uri == DIFFSL_DOCS_URI));
        assert!(resources.iter().any(|resource| resource.uri == created.uri));

        let docs = client
            .peer()
            .read_resource(ReadResourceRequestParams::new(DIFFSL_DOCS_URI))
            .await?;
        let docs_text = read_text(&docs.contents);
        assert!(docs_text.contains("# DiffSL DSL Reference"));
        assert!(docs_text.contains("```pest"));

        let collection = client
            .peer()
            .read_resource(ReadResourceRequestParams::new(PROBLEMS_URI))
            .await?;
        let summaries: Vec<ProblemSummary> =
            serde_json::from_str(&read_text(&collection.contents))?;
        assert_eq!(summaries.len(), 1);
        assert_eq!(summaries[0].id, created.id);

        let problem_doc = client
            .peer()
            .read_resource(ReadResourceRequestParams::new(created.uri.clone()))
            .await?;
        let document: ProblemDocument = serde_json::from_str(&read_text(&problem_doc.contents))?;
        assert_eq!(document.code, LOGISTIC_DIFFSL);
        assert_eq!(document.effective_config.rtol, 1e-8);
        assert_eq!(document.effective_config.atol, 1e-8);

        let solve_request = SolveProblemRequest {
            problem_id: created.id.clone(),
            ode_solver: OdeSolverType::Bdf,
            linear_solver: LinearSolverType::Default,
            params: Vec::new(),
            final_time: None,
            t_eval: Some(vec![0.25, 0.5, 1.0]),
        };

        let solved: SolveProblemResponse = client
            .peer()
            .call_tool(
                CallToolRequestParams::new("solve_problem")
                    .with_arguments(tool_args(&solve_request)),
            )
            .await?
            .into_typed()?;

        assert_eq!(solved.problem_id, created.id);
        assert_eq!(solved.ts.len(), 3);
        assert_eq!(solved.ys.len(), 1);
        assert_eq!(solved.ys[0].len(), 3);
        assert!(!solved.current_state.is_empty());

        client.cancel().await?;
        Ok(())
    }
}
