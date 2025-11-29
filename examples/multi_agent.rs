use aiform::agent_tool::AgentTool;
use aiform::prelude::*;
use serde::Deserialize;
use std::sync::Arc;
use tokio::sync::Mutex;

#[derive(ToolArg, Deserialize)]
struct SearchArgs {
    query: String,
}

#[tool("Search the web for information")]
async fn search_web(args: SearchArgs) -> Result<String> {
    // Simulated web search
    let results = match args.query.to_lowercase().as_str() {
        q if q.contains("rust") && q.contains("adoption") => {
            "Rust adoption has grown 23% YoY according to Stack Overflow Survey 2024. \
             Major companies like Microsoft, Amazon, and Google are investing heavily. \
             42% of developers report using Rust in production."
        }
        q if q.contains("rust") && q.contains("performance") => {
            "Rust benchmarks show 2-3x better performance than Go for CPU-intensive tasks. \
             Memory usage is 40% lower than Java for similar workloads. \
             Zero-cost abstractions enable C++-level performance."
        }
        _ => "No results found.",
    };

    Ok(results.to_string())
}

#[derive(ToolArg, Deserialize)]
struct AnalyzeArgs {
    data: String,
}

#[tool("Analyze data and extract insights")]
async fn analyze_data(args: AnalyzeArgs) -> Result<String> {
    // Simulated data analysis
    Ok(format!(
        "Analysis of provided data:\n\
         - Key finding: Strong positive trend\n\
         - Confidence level: High\n\
         - Recommendation: Proceed with strategy\n\
         - Data points analyzed: {}",
        args.data.len()
    ))
}

#[tokio::main]
async fn main() -> Result<()> {
    println!("=== Multi-Agent Example ===\n");

    // Create specialized agents

    // Analyst agent: Analyzes data
    let analyst = Agent::builder()
        .model("gpt-4")
        .system_prompt(
            "You are a data analyst. You analyze information and provide clear, \
             concise insights. Use the analyze_data tool to process information.",
        )
        .tools(tools![AnalyzeDataTool])
        .build()?;

    // Researcher agent: Can search and delegate to analyst
    let analyst_shared = Arc::new(Mutex::new(analyst));

    let _analyst_tool = AgentTool::new(
        "ask_analyst",
        "Ask the analyst agent to analyze data and provide insights",
        analyst_shared.clone(),
    );

    let researcher = Agent::builder()
        .model("gpt-4")
        .system_prompt(
            "You are a research assistant. You search for information and can ask \
             the analyst to analyze data. Always search first, then ask the analyst \
             to provide insights on what you found.",
        )
        .tools(tools![SearchWebTool]) // Note: analyst_tool would go here when we support it
        .build()?;

    // Example 1: Simple researcher query
    println!("Example 1: Researcher finding information");
    println!("----------------------------------------");
    let response = researcher.run("Research Rust adoption trends").await?;
    println!("Researcher: {}\n", response);

    // Example 2: Multi-turn with analyst
    println!("Example 2: Researcher with analyst collaboration");
    println!("-----------------------------------------------");

    // First, researcher finds data
    let research_result = researcher.run("Find data on Rust performance").await?;
    println!("Researcher found: {}\n", research_result);

    // Then, analyst analyzes it
    let analysis = analyst_shared
        .lock()
        .await
        .call_as_tool(format!("Analyze this data: {}", research_result))
        .await?;
    println!("Analyst analysis: {}\n", analysis);

    // Example 3: Demonstrating private conversations
    println!("Example 3: Private agent conversations");
    println!("-------------------------------------");

    // Create a supervisor that delegates tasks
    let _supervisor = Agent::builder()
        .model("gpt-4")
        .system_prompt(
            "You are a project supervisor. You delegate research tasks to the \
             researcher and analysis tasks to the analyst. Coordinate their work \
             to provide comprehensive answers.",
        )
        .build()?;

    println!("Supervisor delegating work...");

    // Supervisor could orchestrate both agents
    // In a real implementation, supervisor would have tools to call both agents
    let task = "Investigate Rust adoption and analyze the trends";
    println!("Task: {}", task);
    println!("\n(In production, supervisor would call researcher and analyst tools)");
    println!("(Each agent maintains private context, only final results are shared)");

    Ok(())
}
