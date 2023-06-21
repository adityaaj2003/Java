import java.time.LocalDate;
import java.time.format.DateTimeFormatter;
import java.time.temporal.ChronoUnit;
import java.util.Scanner;
public class DateDifference {

	public static void main(String[] args) {
	  Scanner scanner =new Scanner(System.in);
	  DateTimeFormatter formatter=DateTimeFormatter.ofPattern("dd-MM-yyyy");
	  
	  //Get the first date from the user.
	  System.out.print("Enter the first date (DD-MM-YYYY):");
	  String date1String = scanner.next();
	  LocalDate date1 = LocalDate.parse(date1String, formatter);
	  
	  //Get the first date from the user.
	  System.out.print("Enter the second date (DD-MM-YYYY):");
	  String date2String = scanner.next();
	  LocalDate date2 = LocalDate.parse(date2String, formatter);
	  
	  //Compute the difference in days between the two dates
	  long daysBetween = ChronoUnit.DAYS.between(date1, date2);
	  
	  //Display the result
	  System.out.println("The number of days beteween " + date1String + " and " +date2String + " is " +daysBetween);
	  
	  scanner.close();
	}

}