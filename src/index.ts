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
  // Mensaje inicial del asistente de soporte
  const SYSTEM_TEMPLATE =
    `You are frontline support staff for LangCorp, a company that sells computers.
Be concise in your responses.
You can chat with customers and help them with basic questions, but if the customer is having a billing or technical problem,
do not try to answer the question directly or gather information.
Instead, immediately transfer them to the billing or technical team by asking the user to hold for a moment.
Otherwise, just respond conversationally. Recuerda que ya has conversado previamente con el usuario y utiliza el historial de mensajes para ofrecer una respuesta coherente y personalizada.`;

  // Se obtiene la respuesta del modelo para la interacción de soporte general
  const supportResponse = await model.invoke([
    { role: "system", content: SYSTEM_TEMPLATE },
    ...state.messages,
  ]);

  // Definimos la "función" que queremos que el modelo invoque para estructurar la respuesta
  const categorizationFunctions = [
    {
      name: "categorize",
      description: "Determines whether the support representative wants to route the user to billing, technical, or just respond conversationally.",
      parameters: {
        type: "object",
        properties: {
          nextRepresentative: {
            type: "string",
            enum: ["BILLING", "TECHNICAL", "RESPOND"],
            description: "Indicates the routing decision: 'BILLING' for billing team, 'TECHNICAL' for technical team, or 'RESPOND' for a conversational response.",
          },
        },
        required: ["nextRepresentative"],
      },
    },
  ];

  const CATEGORIZATION_SYSTEM_TEMPLATE = `You are an expert customer support routing system.
Your job is to detect whether a customer support representative is routing a user to a billing team or a technical team, or if they are just responding conversationally. Recuerda que ya has conversado previamente con el usuario y utiliza el historial de mensajes para ofrecer una respuesta coherente y personalizada.`;

  const CATEGORIZATION_HUMAN_TEMPLATE =
    `The previous conversation is an interaction between a customer support representative and a user.
Extract whether the representative is routing the user to a billing or technical team, or whether they are just responding conversationally.
Return your answer as a JSON object with a single key "nextRepresentative" whose value is one of:
- "BILLING" (if routing to billing),
- "TECHNICAL" (if routing to technical), or
- "RESPOND" (if just responding).`;

  // Se hace la invocación pasando la definición de función y dejando que el modelo decida si la usa
  const categorizationResponse = await model.invoke(
    [
      { role: "system", content: CATEGORIZATION_SYSTEM_TEMPLATE },
      ...state.messages,
      { role: "user", content: CATEGORIZATION_HUMAN_TEMPLATE },
    ],
    {
      functions: categorizationFunctions,
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
  const SYSTEM_TEMPLATE =
    `You are an expert billing support specialist for LangCorp, a company that sells computers.
Help the user to the best of your ability, but be concise in your responses.
You have the ability to authorize refunds, which you can do by transferring the user to another agent who will collect the required information.
If you do, assume the other agent has all necessary information about the customer and their order.
You do not need to ask the user for more information.

Help the user to the best of your ability, but be concise in your responses.`;

  // Ajustamos el historial para que la pregunta del usuario sea el último mensaje
  let trimmedHistory = state.messages;
  if (trimmedHistory.at(-1)._getType() === "ai") {
    trimmedHistory = trimmedHistory.slice(0, -1);
  }

  // Se obtiene la respuesta del representante de facturación
  const billingRepResponse = await model.invoke([
    {
      role: "system",
      content: SYSTEM_TEMPLATE,
    },
    ...trimmedHistory,
  ]);

  // Definimos la función que queremos que el modelo invoque para categorizar la respuesta
  const categorizationFunctions = [
    {
      name: "categorizeBilling",
      description: "Determines whether the billing support representative wants to refund the user or just respond normally.",
      parameters: {
        type: "object",
        properties: {
          nextRepresentative: {
            type: "string",
            enum: ["REFUND", "RESPOND"],
            description:
              "Indicates if the representative wants to refund the user (REFUND) or just respond (RESPOND).",
          },
        },
        required: ["nextRepresentative"],
      },
    },
  ];

  const CATEGORIZATION_SYSTEM_TEMPLATE =
    `Your job is to detect whether a billing support representative wants to refund the user.`;
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
        content: CATEGORIZATION_SYSTEM_TEMPLATE,
      },
      {
        role: "user",
        content: CATEGORIZATION_HUMAN_TEMPLATE,
      },
    ],
    {
      functions: categorizationFunctions,
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
  const SYSTEM_TEMPLATE =
    `You are an expert at diagnosing technical computer issues. You work for a company called LangCorp that sells computers.
Help the user to the best of your ability, but be concise in your responses.`;

  let trimmedHistory = state.messages;
  // Make the user's question the most recent message in the history.
  // This helps small models stay focused.
  if (trimmedHistory.at(-1)._getType() === "ai") {
    trimmedHistory = trimmedHistory.slice(0, -1);
  }

  const response = await model.invoke([
    {
      role: "system",
      content: SYSTEM_TEMPLATE,
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

const conversationalStream = await graph.stream({
  messages: [{
    role: "user",
    content: "Cómo estas? Mi nombre es Alejandro."
  }]
}, {
  configurable: {
    thread_id: "conversational_testing_id"
  }
});

for await (const value of conversationalStream) {
  console.log("---STEP---CONVERSATIONAL---");
  console.log(value);
  console.log("---END STEP---CONVERSATIONAL---");
}
console.log("Memoria #1:", checkpointer);

const newConversationalStream = await graph.stream({
  messages: [{
    role: "user",
    content: "¿Recuerdas mi nombre y la última vez que hablamos?"
  }]
}, {
  configurable: {
    thread_id: "conversational_testing_id"
  }
});

for await (const value of newConversationalStream) {
  console.log("---STEP---CONVERSATIONAL2---");
  console.log(value);
  console.log("---END STEP---CONVERSATIONAL2---");
}

console.log("Memoria #2:", checkpointer);

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
// app.post("/api/chat", async (req, res) => {
//   try {
    
//     const { message, sessionId } = req.body;
//     const threadId = sessionId || uuidv4();

//     console.log("Processing message:", message);

//     const agentOutput = await graph.invoke(
//       {
//         messages: [{ role: "user", content: message }],
//       },
//       {
//         configurable: { thread_id: threadId },
//       }
//     );

//     console.log("Agent output:", agentOutput);

//     // Devuelve la respuesta procesada
//     res.json({ 
//       message: agentOutput.messages[agentOutput.messages.length - 1].content,
//       threadId,
//      });
//   } catch (error) {
//     console.error("Error processing message:", error);
//     res.status(500).json({ error: "Internal server error." });
//   }
// });

// Inicia el servidor
// app.listen(PORT, () => {
//   console.log(`Server is running on http://localhost:${PORT}`);
// });
