import { DynamicTool } from "@langchain/core/tools";

export const categorizationInitialFunction = [
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

export const categorizationBillingFunction = [
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

// 🔧 Herramienta dinámica para categorizar la solicitud de soporte inicial
export const categorizeInitialSupport = new DynamicTool({
  name: "categorize_initial_support",
  description: "Determines whether the support representative should route the user to billing, technical, or respond conversationally.",
  func: async (input: string) => {
    console.log("Input recibido:", input);
    // Busca 'billing' o 'facturación' (insensible a mayúsculas)
    if (/billing|facturación/i.test(input)) {
      return JSON.stringify({ nextRepresentative: "BILLING" });
    } else if (/technical|técnico/i.test(input)) {
      return JSON.stringify({ nextRepresentative: "TECHNICAL" });
    }
    return JSON.stringify({ nextRepresentative: "RESPOND" });
  },
});


// 🔧 Herramienta para categorizar respuestas del representante de facturación
export const categorizeBillingResponse = new DynamicTool({
    name: "categorize_billing_response",
    description: "Determines if the support response should process a refund or not.",
    func: async (input: string) => {
      console.log("Input recibido:", input);
      // Busca 'refund' o 'reembolso' (insensible a mayúsculas)
      if (/refund|reembolso/i.test(input)) {
        return JSON.stringify({ nextRepresentative: "REFUND" });
      }
      return JSON.stringify({ nextRepresentative: "RESPOND" });
    },
  });