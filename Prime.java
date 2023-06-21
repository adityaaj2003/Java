import java.util.Scanner;
public class Prime {
    public static void main(String args[]){
     Scanner S=new Scanner(System.in);
     int n,i,flag=0;
     System.out.println("Enter the value of n:");
     n=S.nextInt();
     for(i=2;i<n;i++)
    {
      if(n%i==0)
      {
        flag=1;
      }
    }
    if(flag==1)
    System.out.println(n+" is not a prime number");
    else
    System.out.println(n+" is a prime number");
    }
    
}
