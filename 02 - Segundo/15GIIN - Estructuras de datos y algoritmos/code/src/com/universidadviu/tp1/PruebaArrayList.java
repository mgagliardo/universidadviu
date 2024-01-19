import java.util.ArrayList;

/** Una clase que simula una casilla de memoria que almacena un entero */
class Casilla_Entera {   
	private int valor = 0;
    public Casilla_Entera() {}; // inicializa en cero a valor
    public Casilla_Entera( int val)
    {     valor = val;   }
    public int leer( )
    {     return valor;     }
     public void escribir( int x )
    {     valor = x;          }
     public String toString( )
    {    return "valor de la casilla = " + valor;   }
     public boolean equals( Object elem )
    {   
         if ( elem == null || getClass() != elem.getClass())
             return false;
         Casilla_Entera elemTemp = ( Casilla_Entera ) elem;   
                   // cast 	ó	conversión de tipos 
                   //(sabemos que elem es Casilla_Entera)
         return    elemTemp.valor == this.valor;
    }
 }

public class PruebaArrayList {

	public static void main( String [ ] args )
	{   
		// Prueba de la clase ArrayList<E>
		
		// declaramos un arrayList vacío de Casilla_Entera
		ArrayList<Casilla_Entera> a = new ArrayList<>();
		
        a.add(new Casilla_Entera(4));
        a.add(new Casilla_Entera(5));
        a.add(new Casilla_Entera(5));
        a.add(null);
        a.add(new Casilla_Entera(10));
        
        // prueba del metodo toArray de ArrayList (note el CAST porque 
        // devuelve arreglo de Object)
	    Object [ ] las_casillas =  a.toArray();
        //	    Imprimir el arreglo de casillas:
	    for (int i=0; i<las_casillas.length ; i++) {
	    	System.out.println("Casilla numero: "+i+"  "+
	    			           (Casilla_Entera)las_casillas[i]);
	    }
	    
        // prueba del método contains de ArrayList
        Casilla_Entera casilla = new Casilla_Entera(5);
        if (a.contains(casilla)) System.out.println("la contiene");
        else System.out.println("no la contiene");  
        
        // prueba del método get de ArrayList  
        if (a.get(1).equals(casilla)) System.out.println("son iguales");
        else System.out.println("no son iguales");
        
        // prueba del método remove de ArrayList        
	    a.remove(casilla);
	    
        // prueba de toArray e imprimir el arreglo (note el CAST)
	    las_casillas =  a.toArray();
        //	    Imprimir el arreglo de casillas:
	    for (int i=0; i<las_casillas.length ; i++) {
	    	System.out.println("Casilla numero: "+i+"  "+
	    			           (Casilla_Entera)las_casillas[i]);
	    }
	    	
	}
}
