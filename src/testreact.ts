import dotenv from "dotenv";
import { ChatTogetherAI } from "@langchain/community/chat_models/togetherai";
import { ChatOpenAI } from "@langchain/openai";
import { Annotation, MessagesAnnotation } from "@langchain/langgraph";
import { z } from "zod";
import { zodToJsonSchema } from "zod-to-json-schema";
import { NodeInterrupt } from "@langchain/langgraph";
import { StateGraph } from "@langchain/langgraph";
import { MemorySaver } from "@langchain/langgraph";
import { HumanMessage, AIMessage, SystemMessage } from "@langchain/core/messages";
import { createReactAgent } from "@langchain/langgraph/prebuilt";
import { RunnableConfig } from "@langchain/core/runnables";

dotenv.config();

const StateAnnotation = Annotation.Root({
    ...MessagesAnnotation.spec,
    nextRepresentative: Annotation<string>,
    refundAuthorized: Annotation<boolean>,
  });

const checkpointer = new MemorySaver();

const model = new ChatOpenAI({
  temperature: 0,
  model: "gpt-4o",
  apiKey: process.env.OPENAI_API_KEY,
});

// 🟢 Agente de Soporte Inicial
const initialSupportAgent = createReactAgent({
  llm: model,
  tools: [],
  stateModifier: new SystemMessage(
    `You are frontline support staff for LangCorp, a company that sells computers.
Be concise in your responses.
You can chat with customers and help them with basic questions, but if the customer is having a billing or technical problem,
do not try to answer the question directly or gather information.
Instead, immediately transfer them to the billing or technical team by asking the user to hold for a moment.
Otherwise, just respond conversationally. Recuerda que ya has conversado previamente con el usuario y utiliza el historial de mensajes para ofrecer una respuesta coherente y personalizada.`
  ),
});

// 🟢 Agente de Soporte de Facturación
const billingSupportAgent = createReactAgent({
  llm: model,
  tools: [],
  stateModifier: new SystemMessage(
    "You are an expert billing support specialist for LangCorp. " +
    "You assist users with billing issues and can authorize refunds. " +
    "If a refund is needed, transfer them to the refund processing agent."
  ),
});

// 🟢 Agente de Soporte Técnico
const technicalSupportAgent = createReactAgent({
  llm: model,
  tools: [],
  stateModifier: new SystemMessage(
    "You are an expert at diagnosing technical computer issues for LangCorp. " +
    "Help the user efficiently and concisely."
  ),
});

// 🟢 Agente de Procesamiento de Reembolsos
const refundAgent = createReactAgent({
  llm: model,
  tools: [],
  stateModifier: new SystemMessage(
    "You handle refund requests. If a refund is authorized, process it immediately."
  ),
});

const initialSupportNode = async (
    state: typeof StateAnnotation.State,
    config?: RunnableConfig
  ) => {
    // 1. Se invoca al agente de soporte inicial para obtener la respuesta inicial.
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

    // 2. Se define la función de categorización para determinar el enrutamiento.
    const categorizationFunctions = [
      {
        name: "categorize",
        description:
          "Determines whether the support representative wants to route the user to billing, technical, or just respond conversationally.",
        parameters: {
          type: "object",
          properties: {
            nextRepresentative: {
              type: "string",
              enum: ["BILLING", "TECHNICAL", "RESPOND"],
              description:
                "Indicates the routing decision: 'BILLING' for billing team, 'TECHNICAL' for technical team, or 'RESPOND' for a conversational response.",
            },
          },
          required: ["nextRepresentative"],
        },
      },
    ];
  
    const CATEGORIZATION_SYSTEM_TEMPLATE = `You are an expert customer support routing system.
  Your job is to detect whether a customer support representative is routing a user to a billing team or a technical team, or if they are just responding conversationally. Recuerda que ya has conversado previamente con el usuario y utiliza el historial de mensajes para ofrecer una respuesta coherente y personalizada.`;
  
    const CATEGORIZATION_HUMAN_TEMPLATE = `The previous conversation is an interaction between a customer support representative and a user.
  Extract whether the representative is routing the user to a billing or technical team, or whether they are just responding conversationally.
  Return your answer as a JSON object with a single key "nextRepresentative" whose value is one of:
  - "BILLING" (if routing to billing),
  - "TECHNICAL" (if routing to technical), or
  - "RESPOND" (if just responding).`;
  
    // 3. Se hace el segundo invoke para categorizar la respuesta.
    const categorizationResponse = await model.invoke(
      [
        { role: "system", content: CATEGORIZATION_SYSTEM_TEMPLATE },
        ...state.messages,
        { role: "user", content: CATEGORIZATION_HUMAN_TEMPLATE },
      ],
      {
        functions: categorizationFunctions,
        function_call: "auto",
      }
    );
  
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
  
    // 4. Se retorna el estado actualizado, incluyendo la decisión de enrutamiento.
    return {
      messages: state.messages,
      nextRepresentative: categorizationOutput.nextRepresentative,
    };
  };
  
  // Nodo para el soporte de facturación
  const billingSupportNode = async (
    state: typeof StateAnnotation.State,
    config?: RunnableConfig
  ) => {
    // 1. Invocar al agente de facturación.
    const supportResult = await billingSupportAgent.invoke(state, config);
    // Para este nodo se puede recortar el historial si el último mensaje es de AI.
    let trimmedHistory = state.messages;
    if (trimmedHistory.at(-1)._getType() === "ai") {
      trimmedHistory = trimmedHistory.slice(0, -1);
    }
    // Se añade el último mensaje obtenido.
    const newMessages = [
      ...state.messages,
      supportResult.messages[supportResult.messages.length - 1],
    ];
  
    // 2. Definir la función de categorización para facturación.
    const categorizationFunctions = [
      {
        name: "categorizeBilling",
        description:
          "Determines whether the billing support representative wants to refund the user or just respond normally.",
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
  
    const CATEGORIZATION_SYSTEM_TEMPLATE = `Your job is to detect whether a billing support representative wants to refund the user.`;
  
    const CATEGORIZATION_HUMAN_TEMPLATE = `The following text is a response from a customer support representative.
  Extract whether they want to refund the user or not.
  Return your answer as a JSON object with a single key "nextRepresentative" whose value is:
  - "REFUND" if they want to refund the user,
  - "RESPOND" if they do not want to refund the user.
  
  Here is the text:
  
  <text>
  ${supportResult.messages[supportResult.messages.length - 1].content}
  </text>.`;
  
    // 3. Invocar al modelo para categorizar la respuesta de facturación.
    const categorizationResponse = await model.invoke(
      [
        { role: "system", content: CATEGORIZATION_SYSTEM_TEMPLATE },
        { role: "user", content: CATEGORIZATION_HUMAN_TEMPLATE },
      ],
      {
        functions: categorizationFunctions,
        function_call: "auto",
      }
    );
  
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
  
    // 4. Retornar el estado actualizado, incluyendo la decisión para facturación.
    return {
      messages: newMessages,
      nextRepresentative: categorizationOutput.nextRepresentative,
    };
  };
  
  // Nodo para el soporte técnico (no requiere categorización adicional)
  const technicalSupportNode = async (
    state: typeof StateAnnotation.State,
    config?: RunnableConfig
  ) => {
    const result = await technicalSupportAgent.invoke(state, config);
    return {
      messages: [...state.messages, result.messages[result.messages.length - 1]],
    };
  };
  
let workflow = new StateGraph(StateAnnotation)
  .addNode("initial_support", initialSupportNode)
  .addNode("billing_support", billingSupportNode)
  .addNode("technical_support", technicalSupportNode)
  .addEdge("__start__", "initial_support");

// 📌 Definir las transiciones
workflow = workflow.addConditionalEdges(
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

// 📌 Agregar transiciones para los otros agentes
workflow = workflow
    .addEdge("technical_support", "__end__")
    .addEdge("billing_support", "__end__");

console.log("Added edges!");

// 📌 Compilar el workflow
const graph = workflow.compile({ checkpointer });

const conversationalStream = await graph.stream({
    messages: [{
      role: "user",
      content: "Hola, necesito ayuda con mi computadora."
    }]
  }, {
    configurable: { thread_id: "conversational_testing_id" }
  });
  
  for await (const value of conversationalStream) {
    console.log("---STEP---CONVERSATIONAL---");
    console.log(value);
    console.log("---END STEP---CONVERSATIONAL---");
  }
//   console.log("Memoria #1:", checkpointer);

  const conversationalStream2 = await graph.stream({
    messages: [{
      role: "user",
      content: "Cómo estas? Mi nombre es Alejandro."
    }]
  }, {
    configurable: {
      thread_id: "conversational_testing_id"
    }
  });

  for await (const value of conversationalStream2) {
    console.log("---STEP---CONVERSATIONAL---");
    console.log(value);
    console.log("---END STEP---CONVERSATIONAL---");
  }
//   console.log("Memoria #2:", checkpointer);

  const conversationalStream3 = await graph.stream({
    messages: [{
        role: "user",
        content: "¿Recuerdas mi nombre y la última vez que hablamos?"
        }]
    }, {
        configurable: {
        thread_id: "conversational_testing_id"
        }
    });

    for await (const value of conversationalStream3) {
        console.log("---STEP---CONVERSATIONAL---");
        console.log(value);
        console.log("---END STEP---CONVERSATIONAL---");
    }
    // console.log("Memoria #3:", checkpointer);