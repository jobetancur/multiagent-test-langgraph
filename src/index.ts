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
import { categorizationInitialFunction, categorizationBillingFunction } from "./tools";

// Express
import express from "express";
import bodyParser from "body-parser";
import cors from "cors";

// UUID
import { v4 as uuidv4 } from "uuid";

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

const initialSupport = async (state: typeof StateAnnotation.State) => {

  // Se obtiene la respuesta del modelo para la interacción de soporte general
  const supportResponse = await model.invoke([
    { role: "system", content: INITIAL_SUPPORT_MESSAGES.SYSTEM_TEMPLATE },
    ...state.messages,
  ]);

  // Se hace la invocación pasando la definición de función y dejando que el modelo decida si la usa
  const categorizationResponse = await model.invoke(
    [
      { role: "system", content: INITIAL_SUPPORT_MESSAGES.CATEGORIZATION_SYSTEM_TEMPLATE },
      ...state.messages,
      { role: "user", content: INITIAL_SUPPORT_MESSAGES.CATEGORIZATION_HUMAN_TEMPLATE },
    ],
    {
      functions: categorizationInitialFunction,
      function_call: "auto", // Permite que el modelo invoque la función cuando corresponda
    }
  );

  // Extraemos el resultado estructurado. Dependiendo de la respuesta del modelo, puede venir en:
  // - categorizationResponse.additional_kwargs.function_call.arguments, o
  // - directamente en categorizationResponse.content
  let categorizationOutput;
  if (
    categorizationResponse.additional_kwargs &&
    categorizationResponse.additional_kwargs.function_call
  ) {
    const functionCall = categorizationResponse.additional_kwargs.function_call;
    categorizationOutput = JSON.parse(functionCall.arguments);
  } else {
    categorizationOutput = JSON.parse(categorizationResponse.content as string);
  }

  // Se retorna el mensaje inicial junto con la decisión extraída
  return { 
    messages: [...state.messages, supportResponse], 
    nextRepresentative: categorizationOutput.nextRepresentative 
  };
};

const billingSupport = async (state: typeof StateAnnotation.State) => {

  // Ajustamos el historial para que la pregunta del usuario sea el último mensaje
  let trimmedHistory = state.messages;
  if (trimmedHistory.at(-1)._getType() === "ai") {
    trimmedHistory = trimmedHistory.slice(0, -1);
  }

  // Se obtiene la respuesta del representante de facturación
  const billingRepResponse = await model.invoke([
    {
      role: "system",
      content: BILLING_SUPPORT_MESSAGES.SYSTEM_TEMPLATE,
    },
    ...trimmedHistory,
  ]);

  const CATEGORIZATION_HUMAN_TEMPLATE =
    `The following text is a response from a customer support representative.
Extract whether they want to refund the user or not.
Return your answer as a JSON object with a single key "nextRepresentative" whose value is:
- "REFUND" if they want to refund the user,
- "RESPOND" if they do not want to refund the user.

Here is the text:

<text>
${billingRepResponse.content}
</text>.`;

  // Invocamos el modelo pasando la definición de función para que formatee la respuesta
  const categorizationResponse = await model.invoke(
    [
      {
        role: "system",
        content: BILLING_SUPPORT_MESSAGES.CATEGORIZATION_SYSTEM_TEMPLATE,
      },
      {
        role: "user",
        content: CATEGORIZATION_HUMAN_TEMPLATE,
      },
    ],
    {
      functions: categorizationBillingFunction,
      function_call: "auto",
    }
  );

  // Extraemos la salida formateada. Según la respuesta del modelo, puede venir en `function_call.arguments`
  let categorizationOutput;
  if (
    categorizationResponse.additional_kwargs &&
    categorizationResponse.additional_kwargs.function_call
  ) {
    const functionCall = categorizationResponse.additional_kwargs.function_call;
    categorizationOutput = JSON.parse(functionCall.arguments);
  } else {
    categorizationOutput = JSON.parse(categorizationResponse.content as string);
  }

  return {
    messages: [...state.messages, billingRepResponse],
    nextRepresentative: categorizationOutput.nextRepresentative,
  };
};


const technicalSupport = async (state: typeof StateAnnotation.State) => {

  let trimmedHistory = state.messages;
  // Make the user's question the most recent message in the history.
  // This helps small models stay focused.
  if (trimmedHistory.at(-1)._getType() === "ai") {
    trimmedHistory = trimmedHistory.slice(0, -1);
  }

  const response = await model.invoke([
    {
      role: "system",
      content: TECHNICAL_SUPPORT_MESSAGES.SYSTEM_TEMPLATE,
    },
    ...trimmedHistory,
  ]);

  return {
    messages: [...state.messages, response],
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
  // .addNode("handle_refund", handleRefund)
  .addEdge("__start__", "initial_support");

builder = builder.addConditionalEdges(
  "initial_support",
  async (state: typeof StateAnnotation.State) => {
    if (state.nextRepresentative.includes("BILLING")) {
      return "billing";
    } else if (state.nextRepresentative.includes("TECHNICAL")) {
      return "technical";
    } else {
      return "conversational";
    }
  },
  {
    billing: "billing_support",
    technical: "technical_support",
    conversational: "__end__",
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
