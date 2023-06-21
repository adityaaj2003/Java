import java.util.Scanner;
public class add{
   private int a,b,sum;
public static void main(String args[]){
    Scanner S=new Scanner(System.in);
    add Sum=new add();
    System.out.println("Enter the two numbers");
    Sum.a=S.nextInt();
    Sum.b=S.nextInt();
    Sum.sum=Sum.a+Sum.b;
    System.out.println("Sum of two numbers "+Sum.a+" and "+Sum.b+" is: "+Sum.sum);
    S.close();
}
}