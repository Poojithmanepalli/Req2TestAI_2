const OpenAI = require("openai");
const { initRAG }              = require("./ragService");
const { runExtractionAgent }   = require("../agents/extractionAgent");
const { runAnalysisAgent }     = require("../agents/analysisAgent");
const { runTestGenAgent }      = require("../agents/testGenAgent");
const { runQAAgent }           = require("../agents/qaAgent");

const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

// ---------- ORCHESTRATOR TOOLS ----------
// Each tool represents launching a specialized sub-agent
const ORCHESTRATOR_TOOLS = [
  {
    type: "function",
    function: {
      name: "run_extraction_agent",
      description: "Launch the Extraction Agent — it extracts, filters, and deduplicates requirements from the SRS document. Always call this first.",
      parameters: {
        type: "object",
        properties: { reason: { type: "string", description: "Why you are launching this agent now" } },
        required: ["reason"]
      }
    }
  },
  {
    type: "function",
    function: {
      name: "run_analysis_agent",
      description: "Launch the Analysis Agent — it scores SRS coverage (40-95) and identifies missing requirements. Call after extraction.",
      parameters: {
        type: "object",
        properties: { reason: { type: "string", description: "Why you are launching this agent now" } },
        required: ["reason"]
      }
    }
  },
  {
    type: "function",
    function: {
      name: "run_testgen_agent",
      description: "Launch the Test Generation Agent — it generates RAG-enhanced test cases for all requirements. Call after analysis.",
      parameters: {
        type: "object",
        properties: { reason: { type: "string", description: "Why you are launching this agent now" } },
        required: ["reason"]
      }
    }
  },
  {
    type: "function",
    function: {
      name: "run_qa_agent",
      description: "Launch the QA Agent — it reviews test quality and retries weak requirements. Call after test generation ONLY if more than 20% of requirements have weak coverage.",
      parameters: {
        type: "object",
        properties: { reason: { type: "string", description: "How many requirements are weak and why QA is needed" } },
        required: ["reason"]
      }
    }
  },
  {
    type: "function",
    function: {
      name: "compile_final_report",
      description: "Compile all agent outputs into the final structured report. Always call this last.",
      parameters: {
        type: "object",
        properties: { summary: { type: "string", description: "One-sentence summary of what the pipeline found and any corrective steps taken" } },
        required: ["summary"]
      }
    }
  }
];

// ---------- REPORT COMPILER ----------
function compileFinalReport(results, coverageScore, missingRequirements, similarRequirements, agentsRun, processingStart) {
  const grouped = {};
  results.forEach(r => {
    if (!grouped[r.module]) grouped[r.module] = [];
    grouped[r.module].push(r);
  });

  const modules = Object.keys(grouped).map(m => ({
    module: m,
    count:  grouped[m].length,
    items:  grouped[m]
  }));

  const rtm = results.map(r => ({
    reqId:         r.id,
    module:        r.module,
    priority:      r.priority,
    type:          r.type,
    requirement:   r.requirement,
    testCaseCount: r.testCases.length,
    tcIds:         r.testCases.map((_, i) => `${r.id}_TC${i + 1}`)
  }));

  return {
    message:             "Multi-Agent AI pipeline complete",
    agentsRun,
    processingTime:      `${Date.now() - processingStart} ms`,
    stats: {
      total:         results.length,
      functional:    results.filter(r => r.type === "functional").length,
      nonFunctional: results.filter(r => r.type === "non-functional").length
    },
    modules,
    coverageScore,
    missingRequirements,
    similarRequirements: similarRequirements || [],
    rtm
  };
}

// ---------- ORCHESTRATOR AGENT LOOP ----------
async function runAgent(chunks, sendProgress, processingStart) {
  await initRAG();

  // Shared state — orchestrator owns this, sub-agents write to it via their return values
  const state = {
    chunks,
    requirements:        [],
    similarRequirements: [],
    coverageScore:       0,
    missingRequirements: [],
    results:             [],
    agentsRun:           []
  };

  const messages = [
    {
      role: "system",
      content: `You are an Orchestrator Agent coordinating a multi-agent AI pipeline for SRS (Software Requirements Specification) analysis.

You have 4 specialized sub-agents and a compile step:
  • Extraction Agent  — extracts and deduplicates requirements from the SRS
  • Analysis Agent    — scores coverage (40-95) and identifies missing requirements
  • TestGen Agent     — generates RAG-enhanced test cases for every requirement
  • QA Agent         — reviews test quality and improves weak requirements
  • Compile          — produces the final structured report

Coordination rules:
1. Always start with run_extraction_agent.
2. Then run_analysis_agent.
3. Then run_testgen_agent.
4. After test generation: if weakPercent > 20%, launch run_qa_agent. Otherwise skip it.
5. Always finish with compile_final_report.

Each sub-agent is fully autonomous — launch them in order and act on their reported results. Do not micro-manage their internals.`
    },
    {
      role: "user",
      content: `Orchestrate the full SRS analysis pipeline for a document with ${chunks.length} chunks. Coordinate all agents and produce a complete report.`
    }
  ];

  for (let iteration = 0; iteration < 10; iteration++) {
    const response = await client.chat.completions.create({
      model:       "gpt-4o-mini",
      messages,
      tools:       ORCHESTRATOR_TOOLS,
      tool_choice: "auto"
    });

    const message = response.choices[0].message;
    messages.push(message);

    if (response.choices[0].finish_reason === "stop" || !message.tool_calls?.length) break;

    for (const toolCall of message.tool_calls) {
      const name = toolCall.function.name;
      let feedback;

      // ── EXTRACTION AGENT ──────────────────────────────────────────────────
      if (name === "run_extraction_agent") {
        sendProgress(20, "Extraction Agent: extracting & deduplicating requirements...", "extract");
        const output             = await runExtractionAgent(state.chunks);
        state.requirements       = output.requirements;
        state.similarRequirements = output.similarRequirements;
        state.agentsRun.push("ExtractionAgent");
        feedback = {
          agentCompleted:  "ExtractionAgent",
          requirementsFound: state.requirements.length,
          similarPairs:    state.similarRequirements.length,
          types: {
            functional:    state.requirements.filter(r => r.type === "functional").length,
            nonFunctional: state.requirements.filter(r => r.type === "non-functional").length
          },
          nextStep: "Run run_analysis_agent."
        };

      // ── ANALYSIS AGENT ────────────────────────────────────────────────────
      } else if (name === "run_analysis_agent") {
        sendProgress(45, "Analysis Agent: scoring coverage & identifying gaps...", "coverage");
        const output              = await runAnalysisAgent(state.requirements);
        state.coverageScore       = output.coverageScore;
        state.missingRequirements = output.missingRequirements;
        state.agentsRun.push("AnalysisAgent");
        feedback = {
          agentCompleted: "AnalysisAgent",
          coverageScore:  state.coverageScore,
          gapsFound:      state.missingRequirements.length,
          nextStep:       "Run run_testgen_agent."
        };

      // ── TESTGEN AGENT ─────────────────────────────────────────────────────
      } else if (name === "run_testgen_agent") {
        sendProgress(65, "TestGen Agent: generating RAG-enhanced test cases...", "generate");
        state.results     = await runTestGenAgent(state.requirements);
        state.agentsRun.push("TestGenAgent");
        const weakCount   = state.results.filter(r => r.testCases.length <= 1).length;
        const weakPct     = Math.round((weakCount / (state.results.length || 1)) * 100);
        feedback = {
          agentCompleted:   "TestGenAgent",
          testCasesCreated: state.results.reduce((s, r) => s + r.testCases.length, 0),
          modulesFound:     [...new Set(state.results.map(r => r.module))].length,
          weakRequirements: weakCount,
          weakPercent:      `${weakPct}%`,
          nextStep:         weakPct > 20
            ? `${weakPct}% requirements are weak — run run_qa_agent to improve them.`
            : "Quality is good — proceed to compile_final_report."
        };

      // ── QA AGENT ──────────────────────────────────────────────────────────
      } else if (name === "run_qa_agent") {
        const weakCount = state.results.filter(r => r.testCases.length <= 1).length;
        sendProgress(80, `QA Agent: reviewing and improving ${weakCount} weak requirements...`, "retry");
        state.results = await runQAAgent(state.results);
        state.agentsRun.push("QAAgent");
        const stillWeak = state.results.filter(r => r.testCases.length <= 1).length;
        feedback = {
          agentCompleted: "QAAgent",
          improved:       weakCount - stillWeak,
          stillWeak,
          nextStep:       "Proceed to compile_final_report."
        };

      // ── COMPILE ───────────────────────────────────────────────────────────
      } else if (name === "compile_final_report") {
        sendProgress(90, "Compiling final multi-agent report...", "compile");
        const report = compileFinalReport(
          state.results,
          state.coverageScore,
          state.missingRequirements,
          state.similarRequirements,
          state.agentsRun,
          processingStart
        );
        messages.push({
          role:         "tool",
          tool_call_id: toolCall.id,
          content:      JSON.stringify({ status: "report compiled successfully" })
        });
        return report;
      }

      messages.push({
        role:         "tool",
        tool_call_id: toolCall.id,
        content:      JSON.stringify(feedback)
      });
    }
  }

  // Safety fallback — compile whatever state we have
  return compileFinalReport(
    state.results,
    state.coverageScore,
    state.missingRequirements,
    state.similarRequirements,
    state.agentsRun,
    processingStart
  );
}

module.exports = { runAgent };
