package com.universidadviu.tp1;

class FindMaxDemo {

    // public static Comparable findMax( Comparable [ ] a )
    // public static <AnyType> AnyType findMax(AnyType[] a) {
    //     int maxIndex = 0;
    //     for(int i = 1; i < a.length; i++) {
    //         if(a[i].compareTo(a[maxIndex]) > 0) {
    //             maxIndex = i;
    //         }
    //     }
    //     return a[maxIndex];
    // }


    // Integer [] aaa = {1,2,6,8};
    // int aa = A.findMax(aaa);
    // System.out.println( "max: " + aa );

    // public static <AnyType extends Comparable<AnyType>> AnyType findMax(AnyType[] a) {
    public static String findMax(String[] a) {
        int maxIndex = 0;
        for( int i = 1; i < a.length; i++ ) {
            if(a[i].compareTo(a[maxIndex]) > 0 ) {
                maxIndex = i;
            }
        }    
        return a[maxIndex];    
    }

    public static Double findMax(Double[] a) {
        int maxIndex = 0;
        for( int i = 1; i < a.length; i++ ) {
            if(a[i].compareTo(a[maxIndex]) > 0 ) {
                maxIndex = i;
            }
        }    
        return a[maxIndex];    
    }


        /**
     * Test findMax on Shape and String objects.
     * (Integer is discussed in Chapter 4).
     */
    public static void main( String [ ] args ) {
        // Shape [] sh1 = { new Circle(  2.0 ),
                        // new Square(  3.0 ),
                        // new Rectangle( 3.0, 4.0 ) };
        
        String [ ] st1 = { "Joe", "Bob", "Bill", "Zeke" };
        
        // System.out.println( findMax( sh1 ) );
        System.out.println( findMax( st1 ) );
    }

}

class TestClass {
    public static void main(String[] args) {
        Double[] aaa = {1.2, 2.3, 6.4, 8.5};
        Double aa = FindMaxDemo.findMax(aaa);
        System.out.println("ma: " + aa);
    }
}
// ¿Qué está errado, ya sea en el código anterior o en el método findMax, y por qué? Justifique su respuesta. (1 punto)

