const OpenAI = require("openai");
const { coverageAI, missingRequirementsAI } = require("../services/llmService");

const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

const TOOLS = [
  {
    type: "function",
    function: {
      name: "analyze_coverage",
      description: "Score how well the requirements cover the system's needs on a 40-95 scale.",
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
      name: "identify_gaps",
      description: "Identify important requirements that are missing from the SRS.",
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
      name: "finalize_analysis",
      description: "Signal that analysis is complete.",
      parameters: {
        type: "object",
        properties: { summary: { type: "string" } },
        required: ["summary"]
      }
    }
  }
];

async function runAnalysisAgent(requirements) {
  const state = { coverageScore: 0, missingRequirements: [] };

  const messages = [
    {
      role: "system",
      content: `You are a specialized Analysis Agent. Your job is to assess how well the extracted requirements cover the system and identify gaps.

Steps:
1. Call analyze_coverage AND identify_gaps (call both — they can be called in any order).
2. Call finalize_analysis once both are done.

Be efficient. Both tools are independent so call them decisively.`
    },
    {
      role: "user",
      content: `Analyze coverage and identify gaps for ${requirements.length} extracted requirements.`
    }
  ];

  for (let i = 0; i < 6; i++) {
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

      if (name === "analyze_coverage") {
        const score = await coverageAI(requirements);
        state.coverageScore = score || (() => {
          const f  = requirements.filter(r => r.type === "functional").length;
          const nf = requirements.filter(r => r.type === "non-functional").length;
          return Math.min(90, Math.min(50, f * 4) + Math.min(30, nf * 5) + (requirements.length >= 20 ? 10 : 5));
        })();
        feedback = { status: "done", coverageScore: state.coverageScore };

      } else if (name === "identify_gaps") {
        const rawMissing = await missingRequirementsAI(requirements);
        state.missingRequirements = (Array.isArray(rawMissing) ? rawMissing : [])
          .map(r => typeof r === "string" ? { text: r } : r)
          .filter(r => r && r.text && r.text.trim().length > 0)
          .slice(0, 8);
        feedback = { status: "done", gapsFound: state.missingRequirements.length };

      } else if (name === "finalize_analysis") {
        messages.push({
          role: "tool",
          tool_call_id: toolCall.id,
          content: JSON.stringify({ status: "finalized" })
        });
        return { coverageScore: state.coverageScore, missingRequirements: state.missingRequirements };
      }

      messages.push({
        role: "tool",
        tool_call_id: toolCall.id,
        content: JSON.stringify(feedback)
      });
    }
  }

  return { coverageScore: state.coverageScore, missingRequirements: state.missingRequirements };
}

module.exports = { runAnalysisAgent };
