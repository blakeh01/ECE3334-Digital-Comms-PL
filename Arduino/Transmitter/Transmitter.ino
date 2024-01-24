
#include "FspTimer.h"
const int N = 256;
const int data = 15;
const int table[4] = {128,256,512,1024};

const int I = table[data%4];
const int Q = table[data>>2];



// minimum value for T: 12
// with digitalWrite: 13 --> 300 Hz
// reducing analogWriteResolution does not help
const int T = 20;  // 1E6 / (N * T) = 195 Hz
int sineTab[N];
volatile byte index = 0; // automatic wrap around after 255
int v;

static FspTimer fsp_timer;
static void timer_callback([[maybe_unused]]timer_callback_args_t *arg)
{
  uint16_t v = sineTab[index]; 
  index = (index+1)%N;
  analogWrite(DAC, v);
}



void setup() {
  Serial.begin(9600);
  Serial.println("started.");
  analogWriteResolution(12);  // set the analog output resolution to 12 bit (4096 levels)
  for (int i = 0; i < N; i++)
    sineTab[i] = 2046*cos(TWO_PI * i / N)+2046;
  uint8_t timer_type;
  int8_t timer_ch = FspTimer::get_available_timer(timer_type);
  if (timer_ch < 0) {
    Serial.println("get_available_timer() failed.");
    return;
  }
  fsp_timer.begin(TIMER_MODE_PERIODIC, timer_type, static_cast<uint8_t>(timer_ch), 1000.0,100, timer_callback, nullptr);
  fsp_timer.setup_overflow_irq();
  fsp_timer.open();
  fsp_timer.start();
  pinMode(8, OUTPUT);
}

void loop() 
{
  delay(3000);
  fsp_timer.stop();
  analogWrite(DAC, 0);
  for (int i = 0; i < N; i++)
    sineTab[i] = I *sin(TWO_PI * i / N)+Q*cos(TWO_PI * i / N)+2046; 
  delay(500);
  fsp_timer.start();
  Serial.print("I: ");
  Serial.print(I);
  Serial.print(" Q: ");
  Serial.println(Q);
  while(1){}
}
