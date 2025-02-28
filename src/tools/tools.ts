import { tool } from "@langchain/core/tools";
import { z } from "zod";
import { contactCustomerService, validateCity } from "../functions/functions";

export const contactTool = tool(
    async () => {
      const contact = contactCustomerService();
      return contact;
    },
    {
      name: 'contacto_servicio_cliente',
      description: 'Brinda el canal de contacto para ventas y servicios.',
      schema: z.object({}),
    }
);

export const validateCityTool = tool(
    async ({ city }: { city: string }) => {
      const cityValidation = validateCity(city);
      return cityValidation;
    },
    {
      name: "validate_city",
      description: "Valida si la ciudad ingresada por el cliente es una ciudad en la que Servicios y Asesorías tiene presencia. Si la ciudad no esta en los departamentos de Antioquia, Córdoba, Chocó, Norte de Santander, Guainía, Boyacá o Arauca, redirige al cliente a la línea de atención correspondiente. Si la ciudad es pertenece a uno de estos departamentos, pregunta si desea agendar una reunión.",
      schema: z.object({
        city: z.string(),
      }),
    }
);