import java.util.Scanner;
import java.lang.Math;

public class triangle {
    public static void main(String[] args) {
        System.out.println("triangle:");
        Scanner cin=new Scanner(System.in);
        while(cin.hasNextDouble())
        {
            double lenOfSide=cin.nextFloat();
            double len=cin.nextFloat();
            double area=Math.sqrt(3)*lenOfSide*lenOfSide/4;
            double volum=area*len;
            System.out.println("the area is "+String.format("%.2f",area));
            System.out.println("the volum of the Triangle prism is "+String.format("%.2f",volum));



        }

    }
}
