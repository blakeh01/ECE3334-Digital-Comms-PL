
#include "FspTimer.h"
const int N = 256;
const int I = 1024;
const int Q = 256;
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
  Serial.println(v); 
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
  delay(10000);
  fsp_timer.stop();
  analogWrite(DAC, 0);
  for (int i = 0; i < N; i++)
    sineTab[i] = I *sin(TWO_PI * i / N)+Q*cos(TWO_PI * i / N)+2046; 
  delay(1000);
  fsp_timer.start();
  while(1){}
}
