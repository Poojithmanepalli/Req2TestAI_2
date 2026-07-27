const OpenAI = require("openai");
const { extractRequirementsAI, semanticDeduplicateAI } = require("../services/llmService");
const { deduplicateRequirements } = require("../utils/deduplicate");
const { isValidRequirement }      = require("../utils/requirementFilter");
const { findSimilarRequirements } = require("../services/ragService");

const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

function makeFinerChunks(chunks) {
  const finer = [];
  for (const chunk of chunks) {
    const words = chunk.split(" ");
    if (words.length > 120) {
      const mid = Math.floor(words.length / 2);
      finer.push(words.slice(0, mid).join(" "));
      finer.push(words.slice(mid).join(" "));
    } else {
      finer.push(chunk);
    }
  }
  return finer;
}

async function extractAndClean(chunks) {
  const chunkResults = await Promise.all(chunks.map(c => extractRequirementsAI(c)));
  let requirements = [];
  chunkResults.forEach(ai => {
    if (Array.isArray(ai)) {
      requirements.push(...ai.map(r => ({ text: r.text.trim(), type: r.type || "functional" })));
    }
  });
  requirements = requirements.filter(r => isValidRequirement(r.text));
  requirements = deduplicateRequirements(requirements);
  requirements = await semanticDeduplicateAI(requirements);
  return requirements.map((r, i) => ({ id: `REQ_${i + 1}`, text: r.text, type: r.type }));
}

const TOOLS = [
  {
    type: "function",
    function: {
      name: "extract_requirements",
      description: "Extract, filter, and deduplicate requirements from the SRS chunks. Call this first.",
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
      name: "reextract_with_finer_chunks",
      description: "Split chunks into smaller pieces and re-extract. Call ONLY ONCE if fewer than 5 requirements were found.",
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
      name: "finalize_extraction",
      description: "Signal extraction is complete and return the final requirements list.",
      parameters: {
        type: "object",
        properties: { summary: { type: "string" } },
        required: ["summary"]
      }
    }
  }
];

async function runExtractionAgent(chunks) {
  const state = { chunks, requirements: [], hasReextracted: false };

  const messages = [
    {
      role: "system",
      content: `You are a specialized Extraction Agent. Your only job is to extract high-quality software requirements from SRS document chunks.

Steps:
1. Call extract_requirements.
2. If fewer than 5 requirements found, call reextract_with_finer_chunks ONCE to recover missed ones.
3. Call finalize_extraction when done.

Be decisive. Do not loop unnecessarily.`
    },
    {
      role: "user",
      content: `Extract requirements from the SRS document (${chunks.length} chunks).`
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

      if (name === "extract_requirements") {
        state.requirements = await extractAndClean(state.chunks);
        feedback = {
          status: "done",
          total: state.requirements.length,
          agentNote: state.requirements.length < 5
            ? "Very few requirements found. Call reextract_with_finer_chunks."
            : "Extraction complete. Call finalize_extraction."
        };

      } else if (name === "reextract_with_finer_chunks") {
        if (state.hasReextracted) {
          feedback = { status: "skipped", reason: "Already reextracted once. Call finalize_extraction." };
        } else {
          state.hasReextracted = true;
          const finerChunks    = makeFinerChunks(state.chunks);
          const newReqs        = await extractAndClean(finerChunks);
          const existingTexts  = new Set(state.requirements.map(r => r.text.toLowerCase()));
          const added          = newReqs.filter(r => !existingTexts.has(r.text.toLowerCase()));
          const merged         = [...state.requirements, ...added];
          state.requirements   = merged.map((r, i) => ({ ...r, id: `REQ_${i + 1}` }));
          feedback = {
            status: "done",
            newlyFound: added.length,
            totalNow: state.requirements.length,
            agentNote: "Reextraction done. Call finalize_extraction."
          };
        }

      } else if (name === "finalize_extraction") {
        const similarRequirements = await findSimilarRequirements(state.requirements);
        messages.push({
          role: "tool",
          tool_call_id: toolCall.id,
          content: JSON.stringify({ status: "finalized", total: state.requirements.length })
        });
        return { requirements: state.requirements, similarRequirements };
      }

      messages.push({
        role: "tool",
        tool_call_id: toolCall.id,
        content: JSON.stringify(feedback)
      });
    }
  }

  // Fallback: return whatever was extracted
  const similarRequirements = await findSimilarRequirements(state.requirements);
  return { requirements: state.requirements, similarRequirements };
}

module.exports = { runExtractionAgent };
