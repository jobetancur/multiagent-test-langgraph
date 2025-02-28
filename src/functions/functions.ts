import colombia from '../data/colombia.json';

export function contactCustomerService() {
    const customerServiceData = {
      whatsapp: "https://wa.me/573335655669",
      description: "Linea de atención especializada para ventas.",
    };
    console.log('contactCustomerService executed');
    return JSON.stringify(customerServiceData);
}

// Función para eliminar tildes y diéresis
function removeAccents(str: string): string {
    return str.normalize("NFD").replace(/[\u0300-\u036f]/g, "");
  }
  
  // Lista de departamentos permitidos
  const allowedDepartments = [
    "Antioquia",
    "Córdoba",
    "Chocó",
    "Norte de Santander",
    "Guainía",
    "Boyacá",
    "Arauca"
  ];
  
  // Función para validar si el municipio ingresado pertenece a Antioquia, Córdoba, Chocó, Norte de Santander, Guainía, Boyacá o Arauca. Leerlo del archivo colombia.json
  export function validateCity(city: string): string {

    console.log('validateCity executed');

    const normalizedCity = removeAccents(city.toLowerCase());
  
    const filteredDepartments = colombia.filter((dept) =>
      allowedDepartments.includes(dept.departamento)
    );
  
    const cityExists = filteredDepartments.some((dept) =>
      dept.ciudades.some((c) => removeAccents(c.toLowerCase()) === normalizedCity)
    );
  
    if (cityExists) {
      return "Perfecto, tu ciudad está dentro de nuestra cobertura.";
    }
    
    return "Lo siento, actualmente no tenemos cobertura en tu ciudad. Puedes comunicarte en el siguiente enlace: https://wa.me/573186925681";
}