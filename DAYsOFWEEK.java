import java.util.Scanner;
public class DAYsOFWEEK {
    int date,month,year,day[];
    public void countDays()
    {
       System.out.println("Enter first date,month,year respectively");
       Scanner Scan = new Scanner(System.in);
       DAYsOFWEEK d1 = new DAYsOFWEEK();
       DAYsOFWEEK d2 = new DAYsOFWEEK();
       d1.date=Scan.nextInt();
       d1.month=Scan.nextInt();
       d1.year=Scan.nextInt();
       System.out.println("Enter second date,month,year respectively");
       d2.date=Scan.nextInt();
       d2.month=Scan.nextInt();
       d2.year=Scan.nextInt();
       Calender c = Calender.getInstance();
       c.set
    }
     public static void main(String[] args){
        DAYsOFWEEK d3 = new DAYsOFWEEK();
        d3.countDays();
     }
}
