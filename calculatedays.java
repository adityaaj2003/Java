import java.time.DayOfWeek;
import java.time.LocalDate;
import java.time.temporal.ChronoUnit;
import java.util.Scanner;
import java.time.format.DateTimeFormatter;

public class calculatedays {
    public static void main(String[] args) {
            Scanner scanner = new Scanner(System.in);
            DateTimeFormatter formatter = DateTimeFormatter.ofPattern("DD-MM-YYYY");

            System.out.print("enter the first date(DD-MM-YYYY)");
            String date1String = scanner.next();
            LocalDate date1 = LocalDate.parse(date1String,formatter);

            System.out.print("enter the second date(DD-MM-YYYY)");
            String date2String = scanner.next();
            LocalDate date2 = LocalDate.parse(date2String,formatter);

      // LocalDate startDate = LocalDate.of(2023, 1, 1);
            //LocalDate endDate = LocalDate.of(2023, 2, 28);

            int mondays = countWeekdays(date1, date2, DayOfWeek.MONDAY);
            int tuesdays = countWeekdays(date1, date2, DayOfWeek.TUESDAY);
            int wednesdays = countWeekdays(date1, date2, DayOfWeek.WEDNESDAY);
            int thursdays = countWeekdays(date1, date2, DayOfWeek.THURSDAY);
            int fridays = countWeekdays(date1, date2, DayOfWeek.FRIDAY);
            int saturdays = countWeekdays(date1, date2, DayOfWeek.SATURDAY);
            int sundays = countWeekdays(date1, date2, DayOfWeek.SUNDAY);
            


            System.out.println("Number of Mondays: " + mondays);
            System.out.println("Number of Tuesdays: " + tuesdays);
            System.out.println("Number of Wednesdays: " + wednesdays);
            System.out.println("Number of Thursdays: " + thursdays);
            System.out.println("Number of Fridays: " + fridays);
            System.out.println("Number of Saturdays: " + saturdays);
            System.out.println("Number of Sundays: " + sundays);

            scanner.close();
        
    }

    public static int countWeekdays(LocalDate date1, LocalDate date2, DayOfWeek dayOfWeek) {
        int count = 0;
        LocalDate date = date1;

        while (!date.isAfter(date2)) {
            if (date.getDayOfWeek() == dayOfWeek) {
                count++;
            }
            date = date.plus(1, ChronoUnit.DAYS);
        }

        return count;

    }
      
}
