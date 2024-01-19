package com.universidadviu.tp4;

import java.util.Comparator;

class OrderPersonByName implements Comparator<Person> {
   public int compare( Person p1, Person p2 ) {
        // un comparador por nombre 
        return p1.getName().compareTo(p2.getName());
    }
}


class Person implements Comparable<Person> {

    private String dni;
    private String name;
    private int age;
    private String phone;

    public Person(String dni, String n, int ag, String p) {
        this.dni = dni; name = n; age = ag;  phone = p;
    }

    public String toString() {
        return getDni( ) + " " +getName( ) + " " + getAge( ) + " " + getPhoneNumber( );
    }

    // comparar por DNI
    public int compareTo(Person rhs) {
        return dni.compareTo(rhs.dni);
    }
    
    public final String getDni() {
        return dni;
    }
    
    public final String getName() {
        return name;
    }
    
    public final int getAge() {
        return age;
    }

    public final String getPhoneNumber() {
        return phone;
    }

    public final void setPhoneNumber(String newPhone) {
        phone = newPhone;
    }
}

class Student extends Person {

    private double promedio;

    public Student(String dni, String n, int ag, String p, double promedio) {
        super( dni, n, ag,  p );
        this.promedio = promedio;
    }
    
    public String toString() {
        return super.toString( ) + " Promedio= " + getPromedio();
    }
    
    public double getPromedio() {
        return promedio;
    }
    
}

class Employee extends Person
{
    public Employee( String dni, String n, int ag,  String p, double salario )
    {
        super( dni, n, ag,  p );
        this.salario = salario;
    }
    
    public String toString( )
    {
        return super.toString( ) + " $" + getSalario( );
    }
    
    public double getSalario( )
    {
        return salario;
    }
    
    public void raise( double percentRaise )
    {
        salario *= ( 1 + percentRaise );
    }
    
    private double salario;
}

/// codigo alternativo para Leer de un archivo de texto

////Leer archivo e ir guardando a los estudiantes 
//// y empleados en el árbol

//BufferedReader reader = new BufferedReader(new InputStreamReader(
//	      new FileInputStream("archivo.txt"), "UTF-8"));
//
//line = reader.readLine();
//while (line!=null) {
//	 
//	 System.out.println(line);
//	 
//	 tipo = line.split(" ")[0];
//	 System.out.println(tipo+(tipo.equals("S")));
//	 if (tipo.equals("S")) { 
//		 line = reader.readLine();
// 	 dni = line.split(" ")[0];
//		 nombre = reader.readLine();
//		 line = reader.readLine();
// 	 edad = Integer.parseInt(line.split(" ")[0]);
// 	 line = reader.readLine();
// 	 telf = line.split(" ")[0];
// 	 line = reader.readLine();
// 	 calif_o_salario = Double.parseDouble(line.split(" ")[0]);
// 	 Student est = new Student(dni,nombre,edad,telf,
// 			 calif_o_salario);
// 	 System.out.println("estudiante "+dni+" "+nombre+" "+
// 	                    edad+" "+telf+" "+calif_o_salario);
// 	 t.insert(est);
// 	 
//	 }
//	 if (tipo.equals("E")) {
//		 line = reader.readLine();
// 	 dni = line.split(" ")[0];
//		 nombre = reader.readLine();
//		 line = reader.readLine();
// 	 edad = Integer.parseInt(line.split(" ")[0]);
// 	 line = reader.readLine();
// 	 telf = line.split(" ")[0];
// 	 line = reader.readLine();
// 	 calif_o_salario = Double.parseDouble(line.split(" ")[0]);
// 	 Employee emp = new Employee(dni,nombre,edad,telf,
// 			 calif_o_salario);
// 	 System.out.println("empleado "+dni+" "+nombre+" "+
//                 edad+" "+telf+" "+calif_o_salario);
// 	 t.insert(emp);
//	 }
//	 line = reader.readLine();
//}
//reader.close();

