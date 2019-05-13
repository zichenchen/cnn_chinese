package main;

import java.io.*;

public class gnttodata {

    public static void main(String[] args) throws IOException {
       load();
    }
    public static void load() throws IOException {
        File filein=new File("D:\\下载软件\\HWDB1.1trn_gnt\\1001-c.gnt");

        File fileout=new File("D:\\ideaworkspace\\gntTopng\\files\\4");
        DataInputStream dataInputStream=new DataInputStream(new FileInputStream(filein));
        DataOutputStream dataOutputStream=new DataOutputStream(new FileOutputStream(fileout));
        byte[] bytes=new byte[10];
        int count=0;



        while(dataInputStream.read(bytes)!=-1){
            int size=bytes[0]&0xFF|(bytes[1]&0xFF)<<8|(bytes[2]&0xFF)<<16|(bytes[3]&0xFF)<<24;
            String c= new String(bytes,4,2,"gbk");
            int w=bytes[6]&0xFF|(bytes[7]&0xFF)<<8;
            int h=bytes[8]&0xFF|(bytes[9]&0xFF)<<8;
            if((w*h)!=size-10){
                System.out.println("错误的数据");
            }
            System.out.println("开始写入");
            StringBuffer stringBuffer=new StringBuffer();
            int wpadding=0;//增加wpadding使得宽为28的倍数
            int wstep=0;//宽缩小wstep倍
            int hpadding=0;//增加padding使得长为28的倍数
            int hstep=0;//长缩小step倍
            if(w%28!=0){
                wpadding=28-w%28;
                wstep=w/28+1;
            }else{
                wstep=w/28;
            }
            if(h%28!=0){
                hpadding=28-h%28;
                hstep=h/28+1;
            }else{
                hstep=h/28;
            }
          int[][] ints=new int[h+hpadding][w+wpadding];

            for (int i = 0; i <h; i++) {
                for (int j = 0; j <w; j++) {
                    if(i<h&&j<w){
                        if(dataInputStream.readByte()==-1){
                            ints[i][j]=0;
                        }else{
                            ints[i][j]=1;
                        }
                    }else{
                        ints[i][j]=0;
                    }
                }
            }
            int [][] data=new int[h+hpadding][28];
            int number=0;
            for (int i = 0; i <ints.length ; i++) {
                for (int j = 0; j <ints[0].length ; j++) {
                    if(j==0){
                        number=ints[i][j];
                        continue;
                    }
                    if(j%wstep==0){
                        data[i][j/wstep-1]=number;
                        number=ints[i][j];
                    }else{
                        number=number|ints[i][j];
                    }
                }
            }
            for (int i = 0; i <28 ; i++) {
                for (int j = 0; j <28 ; j++) {
                    int sum=0;
                    for (int k = 0; k <hstep ; k++) {
                        sum|=data[(k+i*hstep)][j];
                    }
                    stringBuffer.append(sum+",");
                }
            }

            stringBuffer.append(c);
            stringBuffer.append("\r\n");
            dataOutputStream.writeUTF(String.valueOf(stringBuffer));
            count++;
            System.out.println("完成第"+count+"条");
        }
        dataInputStream.close();
        dataOutputStream.close();

    }
}
