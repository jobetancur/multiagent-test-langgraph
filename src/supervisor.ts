import dotenv from "dotenv";
import { END, Annotation } from "@langchain/langgraph";
import { BaseMessage } from "@langchain/core/messages";
import { z } from "zod";
import { ChatPromptTemplate, MessagesPlaceholder } from "@langchain/core/prompts";
import { ChatOpenAI } from "@langchain/openai";
import { HumanMessage } from "@langchain/core/messages";
import { RunnableConfig } from "@langchain/core/runnables";
import { createReactAgent } from "@langchain/langgraph/prebuilt";
import { SystemMessage } from "@langchain/core/messages";
import { START, StateGraph } from "@langchain/langgraph";
import { MemorySaver } from "@langchain/langgraph";
import { contactTool, validateCityTool } from "./tools/tools";

dotenv.config();

const checkpointer = new MemorySaver();

const members = ["customer_service", "validate_city"] as const;

// This defines the object that is passed between each node
// in the graph. We will create different nodes for each agent and tool
const AgentState = Annotation.Root({
    messages: Annotation<BaseMessage[]>({
      reducer: (x, y) => x.concat(y),
      default: () => [],
    }),
    // The agent node that last performed work
    next: Annotation<string>({
      reducer: (x, y) => y ?? x ?? END,
      default: () => END,
    }),
});

const systemPrompt =
  "You are a supervisor tasked with managing a conversation between the" +
  " following workers: {members}. Given the following user request," +
  " respond with the worker to act next. Each worker will perform a" +
  " task and respond with their results and status. When finished," +
  " respond with FINISH.";

const options = [END, ...members];

// Define the routing function
const routingTool = {
    name: "route",
    description: "Select the next role.",
    schema: z.object({
      next: z.enum([END, ...members]),
    }),
};

const prompt = ChatPromptTemplate.fromMessages([
    ["system", systemPrompt],
    new MessagesPlaceholder("messages"),
    [
      "human",
      "Given the conversation above, who should act next?" +
      " Or should we FINISH? Select one of: {options}",
    ],
]);

const formattedPrompt = await prompt.partial({
    options: options.join(", "),
    members: members.join(", "),
});

const llm = new ChatOpenAI({
  temperature: 0,
  model: "gpt-4o",
  apiKey: process.env.OPENAI_API_KEY,
});

const supervisorChain = formattedPrompt
  .pipe(llm.bindTools(
    [routingTool],
    {
      tool_choice: "route",
    },
  ))
  // select the first one
  // @ts-ignore
  .pipe((x) => (x.tool_calls[0].args));

const customerServiceAgent = createReactAgent({
    llm,
    tools: [contactTool],
    stateModifier: new SystemMessage("You are a web customer service agent. You can use the contactTool to give the customer sales contact information.")
})
  
const customerServiceNode = async (
    state: typeof AgentState.State,
    config?: RunnableConfig,
  ) => {
    const result = await customerServiceAgent.invoke(state, config);
    const lastMessage = result.messages[result.messages.length - 1];
    return {
      messages: [
        new HumanMessage({ content: lastMessage.content, name: "CustomerService" }),
      ],
    };
};
  
const validateCityAgent = createReactAgent({
  llm,
  tools: [validateCityTool],
  stateModifier: new SystemMessage("You are a city validator. Use the validateCityTool to check if the city is within the service area.")
})
  
const validateCityNode = async (
  state: typeof AgentState.State,
  config?: RunnableConfig,
) => {
  const result = await validateCityAgent.invoke(state, config);
  const lastMessage = result.messages[result.messages.length - 1];
  return {
    messages: [
      new HumanMessage({ content: lastMessage.content, name: "ValidateCity" }),
    ],
  };
};

// 1. Create the graph
const workflow = new StateGraph(AgentState)
  // 2. Add the nodes; these will do the work
  .addNode("customer_service", customerServiceNode)
  .addNode("validate_city", validateCityNode)
  .addNode("supervisor", supervisorChain);
// 3. Define the edges. We will define both regular and conditional ones
// After a worker completes, report to supervisor
members.forEach((member) => {
  workflow.addEdge(member, "supervisor");
});

workflow.addConditionalEdges(
  "supervisor",
  (x: typeof AgentState.State) => x.next,
);

workflow.addEdge(START, "supervisor");

const graph = workflow.compile({
    checkpointer
});

let streamResults = graph.stream(
    {
      messages: [
        new HumanMessage({
          content: "Hola soy Alejandro, quiero saber cómo puedo contactar al servicio al cliente.",
        }),
      ],
    },{
        configurable: { thread_id: "conversational_testing_id" },
    }
);
  
for await (const output of await streamResults) {
  if (!output?.__end__) {
    console.log(output);
    console.log("-- Stream Results #1 --");
  }
}

let streamResults2 = graph.stream(
    {
        messages: [
        new HumanMessage({
            content: "Quiero saber si mi ciudad está dentro de su cobertura. Vivo en Medellín.",
        }),
        ],
    },{
        configurable: { thread_id: "conversational_testing_id" },
    }
);

for await (const output of await streamResults2) {
    if (!output?.__end__) {
      console.log(output);
      console.log("-- Stream Results #2 --");
    }
}

let streamResults3 = graph.stream(
    {
        messages: [
        new HumanMessage({
            content: "¿Cómo es mi nombre?",
        }),
        ],
    },{
        configurable: { thread_id: "conversational_testing_id" },
    }
);

for await (const output of await streamResults3) {
    if (!output?.__end__) {
        console.log(output);
        console.log("-- Stream Results #3 --");
    }
}