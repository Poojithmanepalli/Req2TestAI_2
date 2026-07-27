const OpenAI = require("openai");
const { generateTestCasesBatchRAG, generateTestCasesAI } = require("../services/llmService");
const { classifyModule } = require("../services/moduleClassifier");

const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

function assignPriority(text) {
  const t = text.toLowerCase();
  if (t.includes("must") || t.includes("critical")) return "HIGH";
  if (t.includes("should")) return "MEDIUM";
  return "LOW";
}

async function generateBatch(requirements) {
  const batchSize  = 8;
  const batches    = [];
  for (let i = 0; i < requirements.length; i += batchSize) {
    batches.push(requirements.slice(i, i + batchSize));
  }

  const batchOutputs = await Promise.all(batches.map(b => generateTestCasesBatchRAG(b)));

  const results = [];
  for (let b = 0; b < batches.length; b++) {
    const batch = batches[b];
    const aiMap = batchOutputs[b];

    const items = await Promise.all(batch.map(async (req, j) => {
      const ai = aiMap?.[String(j + 1)];
      let testCases = ai?.testCases;

      if (!testCases || testCases.length === 0) testCases = await generateTestCasesAI(req.text);
      if (!testCases || testCases.length === 0) {
        testCases = [
          { title: "Happy Path",    steps: ["Execute the feature with valid input"], expected: "Feature executes successfully" },
          { title: "Invalid Input", steps: ["Provide invalid or missing input"],     expected: "Appropriate error message displayed" }
        ];
      }

      return {
        id:          req.id,
        module:      ai?.module || classifyModule(req.text),
        requirement: req.text,
        type:        req.type,
        priority:    assignPriority(req.text),
        testCases:   testCases.slice(0, 5)
      };
    }));

    results.push(...items);
  }
  return results;
}

const TOOLS = [
  {
    type: "function",
    function: {
      name: "generate_test_cases",
      description: "Generate RAG-enhanced test cases for all requirements using expert testing patterns from Pinecone.",
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
      name: "finalize_test_generation",
      description: "Signal test case generation is complete.",
      parameters: {
        type: "object",
        properties: { summary: { type: "string" } },
        required: ["summary"]
      }
    }
  }
];

async function runTestGenAgent(requirements) {
  const state = { results: [] };

  const messages = [
    {
      role: "system",
      content: `You are a specialized Test Generation Agent. Your job is to generate comprehensive, RAG-enhanced test cases for every requirement.

Steps:
1. Call generate_test_cases to produce test cases for all requirements.
2. Call finalize_test_generation when done.

Be efficient. One call to generate_test_cases is sufficient.`
    },
    {
      role: "user",
      content: `Generate test cases for ${requirements.length} requirements using RAG-retrieved testing patterns.`
    }
  ];

  for (let i = 0; i < 4; i++) {
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

      if (name === "generate_test_cases") {
        state.results  = await generateBatch(requirements);
        const weakCount = state.results.filter(r => r.testCases.length <= 1).length;
        feedback = {
          status:          "done",
          totalTestCases:  state.results.reduce((s, r) => s + r.testCases.length, 0),
          modulesFound:    [...new Set(state.results.map(r => r.module))].length,
          weakCount,
          agentNote:       "Generation complete. Call finalize_test_generation."
        };

      } else if (name === "finalize_test_generation") {
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

module.exports = { runTestGenAgent };
