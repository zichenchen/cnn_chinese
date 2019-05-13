package main;

import java.io.*;

public class datatotraindata {
    public static void main(String[] args) throws IOException {
        BufferedReader in = new BufferedReader(new FileReader("D:\\ideaworkspace\\gntTopng\\files\\4"));
        BufferedWriter out=new BufferedWriter(new FileWriter("D:\\ideaworkspace\\gntTopng\\files\\2"));
        String line;
        int count=0;
        while ((line = in.readLine()) != null) {
            String[] datas = line.split("，|%");
           if (datas.length == 0)
                continue;
            StringBuffer buffer=new StringBuffer();
            for (int i =1; i < datas.length-1; i++){
               buffer.append(datas[i]+"，");
            }
           buffer.append(count+",");
            buffer.append(datas[datas.length-1]);
            buffer.append("\r\n");
            out.write(String.valueOf(buffer));
            count++;
        }
        in.close();
        out.close();
    }
}
