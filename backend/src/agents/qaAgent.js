const OpenAI = require("openai");
const { generateTestCasesAI } = require("../services/llmService");

const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

const TOOLS = [
  {
    type: "function",
    function: {
      name: "review_test_quality",
      description: "Assess how many requirements have weak test coverage (≤1 test case) and report the quality score.",
      parameters: {
        type: "object",
        properties: { reason: { type: "string" } },
        required: ["reason"]
      }
    }
  },
  {
    type: "function",
    function: {
      name: "retry_weak_requirements",
      description: "Retry test case generation for requirements with weak coverage. Call ONLY ONCE if more than 20% of requirements are weak.",
      parameters: {
        type: "object",
        properties: { reason: { type: "string" } },
        required: ["reason"]
      }
    }
  },
  {
    type: "function",
    function: {
      name: "finalize_qa",
      description: "Signal QA review is complete.",
      parameters: {
        type: "object",
        properties: { summary: { type: "string" } },
        required: ["summary"]
      }
    }
  }
];

async function runQAAgent(results) {
  const state = { results, hasRetried: false };

  const weakCount = results.filter(r => r.testCases.length <= 1).length;
  const weakPct   = Math.round((weakCount / results.length) * 100);

  const messages = [
    {
      role: "system",
      content: `You are a specialized QA Agent. Your job is to review test case quality and improve weak coverage.

Steps:
1. Call review_test_quality to assess the current state.
2. If more than 20% of requirements are weak, call retry_weak_requirements ONCE.
3. Call finalize_qa when done.

Be decisive. Only retry if truly needed.`
    },
    {
      role: "user",
      content: `Review quality for ${results.length} requirements. ${weakCount} (${weakPct}%) currently have weak test coverage.`
    }
  ];

  for (let i = 0; i < 5; i++) {
    const response = await client.chat.completions.create({
      model: "gpt-4o-mini",
      messages,
      tools: TOOLS,
      tool_choice: "auto"
    });

    const message = response.choices[0].message;
    messages.push(message);
    if (response.choices[0].finish_reason === "stop" || !message.tool_calls?.length) break;

    for (const toolCall of message.tool_calls) {
      const name = toolCall.function.name;
      let feedback;

      if (name === "review_test_quality") {
        const weak = state.results.filter(r => r.testCases.length <= 1);
        const pct  = Math.round((weak.length / state.results.length) * 100);
        feedback = {
          status:            "done",
          totalRequirements: state.results.length,
          weakRequirements:  weak.length,
          weakPercent:       `${pct}%`,
          agentNote:         pct > 20
            ? `${pct}% requirements are weak. Call retry_weak_requirements.`
            : "Quality is acceptable. Call finalize_qa."
        };

      } else if (name === "retry_weak_requirements") {
        if (state.hasRetried) {
          feedback = { status: "skipped", reason: "Already retried once. Call finalize_qa." };
        } else {
          state.hasRetried     = true;
          const weakItems      = state.results.filter(r => r.testCases.length <= 1);
          const retried        = await Promise.all(
            weakItems.map(async (item) => {
              const newTCs = await generateTestCasesAI(item.requirement);
              if (newTCs && newTCs.length > item.testCases.length) {
                return { ...item, testCases: newTCs.slice(0, 5) };
              }
              return item;
            })
          );
          const improvedMap    = {};
          retried.forEach(item => { improvedMap[item.id] = item; });
          state.results        = state.results.map(r => improvedMap[r.id] || r);
          const stillWeak      = state.results.filter(r => r.testCases.length <= 1).length;
          feedback = {
            status:    "done",
            improved:  weakItems.length - stillWeak,
            stillWeak,
            agentNote: "Retry complete. Call finalize_qa."
          };
        }

      } else if (name === "finalize_qa") {
        messages.push({
          role:         "tool",
          tool_call_id: toolCall.id,
          content:      JSON.stringify({ status: "finalized" })
        });
        return state.results;
      }

      messages.push({
        role:         "tool",
        tool_call_id: toolCall.id,
        content:      JSON.stringify(feedback)
      });
    }
  }

  return state.results;
}

module.exports = { runQAAgent };
