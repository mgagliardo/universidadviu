package com.universidadviu.clase2;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.Scanner;

public class LeerDeArchivoYCalcularEdades {

    public static void main(String[] args) {
        try {
            File myObj = new File("./edades.txt");
            System.out.println(myObj.getAbsolutePath());
            Scanner myReader = new Scanner(myObj);
            while (myReader.hasNextLine()) {
              String data = myReader.nextLine();
              System.out.println(data);
            }
            myReader.close();
          } catch (FileNotFoundException e) {
            System.out.println("An error occurred.");
            e.printStackTrace();
          }
        }

}
