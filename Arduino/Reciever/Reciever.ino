const int pin = A0;
int val = 0 ;
const int freq = 400;
const int sample_size = 1000;
const int zeros = sample_size*.1;

const double symbol_time =2;
const double Fs;
const double Ts = 1/Fs;
const double delay_time = Ts*1000;
const double points = symbol_time/Ts;


int samples [points];
int cose_vec [points];
int prod[points];
void setup() {
  Serial.begin(9600);
  pinMode(A0, INPUT);
}

void loop() {
  //idle time
  while(val<=100){
   val = analogRead(pin);
   delay(delay_time); 
  }
  //reading the cosine signal
  for(int i = 0; i<=points;i++){
    val = analogRead(pin);
    cose_vec[i] = val;
    delay(delay_time);
  }
  zero_sampling();
  //reading the I+Q signal
  for(int i = 0; i<points;i++){
    val = analogRead(pin);
    samples[i] = val;
    delay(delay_time);
  }
  dot_product(prod,cose_vec,samples);
}

void zero_sampling(){
  //wait for a pause
  int counter;
  int threshold = 1000;
  while(1){
    val = analogRead(pin);
    if(val<=100){
      if(counter>=threshold)
        break;
      counter+=1;
    }
    else{
      counter=0;
    }
  }
}

void dot_product(int prod[],int vec1[],int vec2[],int SIZE){
  for(int i =0;i<SIZE;i++)
  {
    prod[i] = vec1[i]*vec2[i];
  }
}
