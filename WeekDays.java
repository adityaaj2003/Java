package termwork;
import java.awt.Color;
import java.awt.Dimension;
import java.awt.Font;
import java.awt.GridLayout;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.time.DayOfWeek;
import java.time.LocalDate;
import java.time.format.DateTimeFormatter;
import java.time.format.DateTimeParseException;
import java.time.temporal.ChronoUnit;
import javax.swing.BorderFactory;
import javax.swing.JButton;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JTextField;
import javax.swing.SwingConstants;

public class WeekDays extends JFrame implements ActionListener{

	private JPanel panel;
	private JLabel nameLabel1;
	private JButton submitButton;
	private JLabel nameLabel2;
	private JTextField nameField1;
	private JTextField nameField2;
	private LocalDate d2 ;
	private LocalDate d1;
	private JLabel outputSun;
	private JLabel outputMon;
	private JLabel outputTue;
	private JLabel outputWed;
	private JLabel outputThu;
	private JLabel outputFri;
	private JLabel outputSat;
	JFrame outputFrame;
	 int sun,mon,tue,wed,thu,fri,sat;
	WeekDays(){
		  panel = new JPanel(new GridLayout(5,5,5,10));
	      this.setSize(500,500);
	      this.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
	      this.getContentPane().setBackground(new Color(0xfffffff));
	      panel.setBorder(BorderFactory.createEmptyBorder(10,10,10,10));
	      panel.setPreferredSize(new Dimension(200,100));
	      nameLabel1 = new JLabel("Enter First date:");
	      nameLabel1.setFont(new Font("Serif", Font.BOLD,18));
	      nameLabel1.setHorizontalAlignment(SwingConstants.CENTER);
	      nameField1 = new JTextField();
	      nameField1.setFont(new Font("Serif",Font.PLAIN,18));
	      nameField1.setPreferredSize(new Dimension(15,30));
	      
	      nameLabel2 = new JLabel("Enter Second date :");
	      nameLabel2.setFont(new Font("Serif", Font.BOLD,18));
	      nameLabel2.setHorizontalAlignment(SwingConstants.CENTER);
	      nameField2 = new JTextField();
	      nameField2.setFont(new Font("Serif",Font.PLAIN,18));
	      nameField2.setPreferredSize(new Dimension(20,30));
	      submitButton = new JButton("Submit");
	      submitButton.addActionListener(this);
	      panel.add(nameLabel1);
	      panel.add(nameField1);
	      panel.add(nameLabel2);
	      panel.add(nameField2);
	      panel.add(submitButton);
	      this.setVisible(true);
	      this.add(panel);
	      
	}
	public void actionPerformed(ActionEvent e) {
		if(e.getSource()==submitButton) {
		  DateTimeFormatter formatter=DateTimeFormatter.ofPattern("dd-MM-yyyy");
		  String d1String= nameField1.getText();
		  String d2String= nameField2.getText();
		  try {
		    d1 = LocalDate.parse(d1String,formatter);
		    d2 = LocalDate.parse(d2String,formatter);
			LocalDate temp = d1;
			long daysBetween = ChronoUnit.DAYS.between(d1, d2);
		    for(int i=0;i<daysBetween;i++)
			{
			 DayOfWeek day =temp.getDayOfWeek();
			 int s = day.getValue();
			switch(s)
			{
			case 1:mon++;
			       break;
			case 2:tue++;
		       break;
			case 3:wed++;
		       break;
			case 4:thu++;
		       break;
			case 5:fri++;
		       break;
			case 6:sat++;
		       break;
			case 7:sun++;
		       break;
			 }
			temp=temp.plusDays(1);
			}
		   
		    outputFrame = new JFrame("Weekday Count");
            outputFrame.setLayout(new GridLayout(7, 1));
            outputFrame.setSize(400, 300);
            outputFrame.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
            outputFrame.getContentPane().setBackground(new Color(0xffffff));
			outputSun = new JLabel("Number of Sundays:    "+sun);
			outputMon = new JLabel("Number of Monndays:   "+mon);
			outputTue = new JLabel("Number of Tuedays:    "+tue);
			outputWed = new JLabel("Number of Wednesdays: "+wed);
			outputThu = new JLabel("Number of Thursdays:  "+thu);
            outputFri = new JLabel("Number of Fridays:    "+fri);
            outputSat = new JLabel("Number of Satdays:    "+sat);
            outputFrame.add(outputSun);
            outputFrame.add(outputMon);
            outputFrame.add(outputTue);
            outputFrame.add(outputWed);
            outputFrame.add(outputThu);
            outputFrame.add(outputFri);
            outputFrame.add(outputSat);
            setVisible(true);
		}
		 catch(DateTimeParseException ex) {
	    	  System.out.println("Invalid date");
	      }
		}
	}

	public static void main(String[] args) {
		
      new WeekDays();
      }
	
}
