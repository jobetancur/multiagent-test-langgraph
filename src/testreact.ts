import dotenv from "dotenv";
import { ChatTogetherAI } from "@langchain/community/chat_models/togetherai";
import { ChatOpenAI } from "@langchain/openai";
import { Annotation, MessagesAnnotation } from "@langchain/langgraph";
import { z } from "zod";
import { zodToJsonSchema } from "zod-to-json-schema";
import { NodeInterrupt } from "@langchain/langgraph";
import { StateGraph } from "@langchain/langgraph";
import { MemorySaver } from "@langchain/langgraph";
import { HumanMessage, AIMessage } from "@langchain/core/messages";
import { INITIAL_SUPPORT_MESSAGES, BILLING_SUPPORT_MESSAGES, TECHNICAL_SUPPORT_MESSAGES } from "./constants";
import { categorizeInitialSupport, categorizeBillingResponse } from "./tools";
import { AgentExecutor, createOpenAIFunctionsAgent } from "langchain/agents";

// Express
import express from "express";
import bodyParser from "body-parser";
import cors from "cors";

// UUID
import { v4 as uuidv4 } from "uuid";
import { ChatPromptTemplate, MessagesPlaceholder } from "@langchain/core/prompts";

const app = express();
const PORT = 3008;

app.use(cors()); // Permite solicitudes desde otros dominios
app.use(bodyParser.json());

dotenv.config();

const checkpointer = new MemorySaver();

const model = new ChatOpenAI({
  temperature: 0,
  model: "gpt-4o",
  apiKey: process.env.OPENAI_API_KEY,
});

const StateAnnotation = Annotation.Root({
  ...MessagesAnnotation.spec,
  nextRepresentative: Annotation<string>,
  refundAuthorized: Annotation<boolean>,
});

// ✅ Definir el prompt correctamente como `ChatPromptTemplate`
const prompt = ChatPromptTemplate.fromMessages([
  ["system", `${INITIAL_SUPPORT_MESSAGES.SYSTEM_TEMPLATE}`],
  ["human", "Tools available: {tool_names}"],
  ["human", "Previous tool outputs: {agent_scratchpad}"],
  new MessagesPlaceholder("chat_history"),
]);

// ✅ Resolver la promesa antes de asignar el agente
const createAgent = async () => {
  const tools = [categorizeInitialSupport]; // Lista de herramientas disponibles
  return await createOpenAIFunctionsAgent({
    llm: model,
    tools: tools,
    prompt: prompt, // 🔥 Pasamos `ChatPromptTemplate`, NO un string formateado
  });
};

// ✅ Configurar `AgentExecutor`
const setupAgentExecutor = async () => {
  const agent = await createAgent(); // 🔥 Resolver la promesa
  return new AgentExecutor({
    agent,
    tools: [categorizeInitialSupport],
    verbose: true,
  });
};

// ✅ Convertir el prompt en un `ChatPromptTemplate`
const billingPrompt = ChatPromptTemplate.fromMessages([
  ["system", `${BILLING_SUPPORT_MESSAGES.SYSTEM_TEMPLATE}`],
  ["human", "Tools available: {tool_names}"],
  ["human", "Previous tool outputs: {agent_scratchpad}"],
  new MessagesPlaceholder("chat_history"),
]);

// ✅ Crear el agente para `billingSupport`
const createBillingAgent = async () => {
  const tools = [categorizeBillingResponse]; // Lista de herramientas disponibles
  return await createOpenAIFunctionsAgent({
    llm: model, // Modelo OpenAI
    tools: tools, // Herramienta de categorización de reembolsos
    prompt: billingPrompt,
  });
};

// ✅ Configurar `AgentExecutor` para facturación
const setupBillingAgentExecutor = async () => {
  const agent = await createBillingAgent(); // 🔥 Resolver promesa
  return new AgentExecutor({
    agent,
    tools: [categorizeBillingResponse], // Pasamos la herramienta
    verbose: true,
  });
};

// ✅ Convertir el prompt en un `ChatPromptTemplate`
const technicalPrompt = ChatPromptTemplate.fromMessages([
  ["system", `${TECHNICAL_SUPPORT_MESSAGES.SYSTEM_TEMPLATE}`],
  ["human", "Tools available: {tool_names}"],
  ["human", "Previous tool outputs: {agent_scratchpad}"],
  new MessagesPlaceholder("chat_history"),
]);

// ✅ Crear el agente para `technicalSupport`
const createTechnicalAgent = async () => {
  const tools: never[] = []; // No necesita herramientas
  return await createOpenAIFunctionsAgent({
    llm: model, // Modelo OpenAI
    tools: tools, // No necesita herramientas
    prompt: technicalPrompt,
  });
};

// ✅ Configurar `AgentExecutor` para soporte técnico
const setupTechnicalAgentExecutor = async () => {
  const agent = await createTechnicalAgent(); // 🔥 Resolver promesa
  return new AgentExecutor({
    agent,
    tools: [], // No hay herramientas en este caso
    verbose: true,
  });
};

// 🔄 Actualizar el `initialSupport` para usar el nuevo `AgentExecutor`
const initialSupport = async (state: typeof StateAnnotation.State) => {
  let trimmedHistory = state.messages;
  if (trimmedHistory.length > 0 && trimmedHistory.at(-1)?._getType() === "ai") {
    trimmedHistory = trimmedHistory.slice(0, -1);
  }

  // ⚡ Esperar a que el AgentExecutor esté listo
  const agentExecutor = await setupAgentExecutor();

  // 🔄 Ejecutar el agente con el historial de mensajes
  const conversationHistory = trimmedHistory.map((msg) => msg.content).join("\n");
  const supportResponse = await agentExecutor.call({
    input: trimmedHistory.map((msg) => msg.content).join("\n"),
    tool_names: "categorizeInitialSupport", // O los nombres de tus herramientas, según corresponda
    agent_scratchpad: "",
    chat_history: conversationHistory,
  });

  // Extraer el resultado estructurado
  let categorizationOutput;
  try {
    categorizationOutput = JSON.parse(supportResponse.output);
  } catch (error) {
    categorizationOutput = { nextRepresentative: "RESPOND" }; // Default en caso de error
  }

  return {
    messages: [...state.messages, { role: "assistant", content: supportResponse.output }],
    nextRepresentative: categorizationOutput.nextRepresentative,
  };
};


// 🔄 `billingSupport` con el nuevo enfoque
const billingSupport = async (state: typeof StateAnnotation.State) => {
  let trimmedHistory = state.messages;
  if (trimmedHistory.length > 0 && trimmedHistory.at(-1)?._getType() === "ai") {
    trimmedHistory = trimmedHistory.slice(0, -1);
  }

  // ⚡ Esperar a que `AgentExecutor` esté listo
  const agentExecutor = await setupBillingAgentExecutor();

  // 🔄 Ejecutar el agente con el historial de mensajes
  const conversationHistory = trimmedHistory.map((msg) => msg.content).join("\n");
  const billingRepResponse = await agentExecutor.call({
    input: trimmedHistory.map((msg) => msg.content).join("\n"),
    tool_names: "categorizeBillingResponse", // O los nombres de tus herramientas, según corresponda
    agent_scratchpad: "",
    chat_history: conversationHistory
  });

  // Extraer la decisión de categorización
  let categorizationOutput;
  try {
    categorizationOutput = JSON.parse(billingRepResponse.output);
  } catch (error) {
    categorizationOutput = { nextRepresentative: "RESPOND" }; // Default en caso de error
  }

  return {
    messages: [...state.messages, { role: "assistant", content: billingRepResponse.output }],
    nextRepresentative: categorizationOutput.nextRepresentative,
  };
};

// 🔄 `technicalSupport` con el nuevo enfoque
const technicalSupport = async (state: typeof StateAnnotation.State) => {
  let trimmedHistory = state.messages;
  if (trimmedHistory.length > 0 && trimmedHistory.at(-1)?._getType() === "ai") {
    trimmedHistory = trimmedHistory.slice(0, -1);
  }

  // ⚡ Esperar a que `AgentExecutor` esté listo
  const agentExecutor = await setupTechnicalAgentExecutor();

  // 🔄 Ejecutar el agente con el historial de mensajes
  const conversationHistory = trimmedHistory.map((msg) => msg.content).join("\n");
  const technicalRepResponse = await agentExecutor.call({
    input: trimmedHistory.map((msg) => msg.content).join("\n"),
    tool_names: "", // No necesita herramientas
    agent_scratchpad: "",
    chat_history: conversationHistory
  });

  return {
    messages: [...state.messages, { role: "assistant", content: technicalRepResponse.output }],
  };
};

const handleRefund = async (state: typeof StateAnnotation.State) => {
  if (!state.refundAuthorized) {
    console.log("--- HUMAN AUTHORIZATION REQUIRED FOR REFUND ---");
    throw new NodeInterrupt("Human authorization required.");
  }
  return {
    messages: {
      role: "assistant",
      content: "Refund processed!",
    },
  };
};

let builder = new StateGraph(StateAnnotation)
  .addNode("initial_support", initialSupport)
  .addNode("billing_support", billingSupport)
  .addNode("technical_support", technicalSupport)
  .addNode("fallback", async (state) => {
    return { messages: [...state.messages, { role: "assistant", content: "Lo siento, no pude clasificar tu solicitud." }] };
  })
  .addEdge("__start__", "initial_support");

builder = builder.addConditionalEdges(
  "initial_support",
  async (state: typeof StateAnnotation.State) => {
    const rep = state.nextRepresentative || "";
    if (rep.includes("BILLING")) return "billing";
    if (rep.includes("TECHNICAL")) return "technical";
    if (rep.includes("RESPOND")) return "conversational";
    return "fallback"; // Fallback si el modelo no devuelve un valor esperado
  },
  {
    billing: "billing_support",
    technical: "technical_support",
    conversational: "__end__",
    fallback: "fallback",
  }
);

console.log("Added edges!");

builder = builder
  .addEdge("technical_support", "__end__")
  .addEdge("billing_support", "__end__");

console.log("Added edges!");

const graph = builder.compile({ 
  checkpointer,
});

// const conversationalStream = await graph.stream({
//   messages: [{
//     role: "user",
//     content: "Cómo estas? Mi nombre es Alejandro."
//   }]
// }, {
//   configurable: {
//     thread_id: "conversational_testing_id"
//   }
// });

// for await (const value of conversationalStream) {
//   console.log("---STEP---CONVERSATIONAL---");
//   console.log(value);
//   console.log("---END STEP---CONVERSATIONAL---");
// }
// console.log("Memoria #1:", checkpointer);

// const newConversationalStream = await graph.stream({
//   messages: [{
//     role: "user",
//     content: "¿Recuerdas mi nombre y la última vez que hablamos?"
//   }]
// }, {
//   configurable: {
//     thread_id: "conversational_testing_id"
//   }
// });

// for await (const value of newConversationalStream) {
//   console.log("---STEP---CONVERSATIONAL2---");
//   console.log(value);
//   console.log("---END STEP---CONVERSATIONAL2---");
// }

// console.log("Memoria #2:", checkpointer);

// const stream = await graph.stream(
//   {
//     messages: [{ role: "user", content: "Hola, ¿cómo estás?" }],
//   },
//   {
//     configurable: { thread_id: "refund_testing_id" },
//   }
// );

// const responses = [];
// for await (const value of stream) {
//   responses.push(value);
// }

// // Buscar el nodo que tiene mensajes
// let agentResponse = null;
// for (const node of responses) {
//   const nodeKey = Object.keys(node)[0]; // Ejemplo: "initial_support"
//   const nodeValue = node[nodeKey];

//   if (nodeValue?.messages?.[0]?.content) {
//     agentResponse = nodeValue.messages[0].content; // Extraer el content
//     break; // Salir del bucle cuando se encuentra el primer mensaje válido
//   }
// }

// // Si no se encontró ninguna respuesta válida
// agentResponse = agentResponse || "No response from any agent";

// console.log("Agent response:", agentResponse);


// Endpoint para procesar mensajes
app.post("/api/chat", async (req, res) => {
  try {
    
    const { message, sessionId } = req.body;
    const threadId = sessionId;

    console.log("Processing message:", message);

    const agentOutput = await graph.invoke(
      {
        messages: [{ role: "user", content: message }],
      },
      {
        configurable: { thread_id: threadId },
      }
    );

    console.log("Agent output:", agentOutput);

    // Devuelve la respuesta procesada
    res.json({ 
      message: agentOutput.messages[agentOutput.messages.length - 1].content,
      threadId,
     });
  } catch (error) {
    console.error("Error processing message:", error);
    res.status(500).json({ error: "Internal server error." });
  }
});

// Inicia el servidor
app.listen(PORT, () => {
  console.log(`Server is running on http://localhost:${PORT}`);
});
