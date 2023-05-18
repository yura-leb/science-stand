/* 
 * ngtcp2 & InOpSy
 *
 * Time Logger class used to sync logs with experiment time.
 */
#ifndef TIME_LOGGER_H
#define TIME_LOGGER_H

#include <thread>
#include <string>
#include <ctime>

#include <stdio.h>

// Defines, how frequent do we write timestamp in log (in ms)
# define INOPSY_TIMER_LOG_PARTITION 10

/**
 * 
 * Next class is needed to optimise data calculation by writing 
 * time periods in log (stderr)
 * 
 */
class TimeLogger
  {
  public:
    void yield_time()
    {
      int32_t time_to_wait = 0;
      uint32_t partition = INOPSY_TIMER_LOG_PARTITION;
      start_clock = get_current_clock_ns();
      next_clock = start_clock;
      uint32_t current_time = 0;
      while (true) {
        current_time += partition;
        fprintf(stderr, "!TSMP_I:%" PRIu32 "\n", current_time); // means timestamp in ms
        time_to_wait = calc_time_to_wait(partition);
        if (time_to_wait > 0) {
          std::this_thread::sleep_for(std::chrono::nanoseconds(time_to_wait));
        }
      }
    }

private:
  uint32_t get_current_clock_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
    std::chrono::system_clock::now().time_since_epoch()).count();
  }

  int32_t calc_time_to_wait(uint32_t partition) {
    next_clock += one_ms_in_ns * partition;
    return next_clock - get_current_clock_ns();
  }

  static constexpr uint32_t one_ms_in_ns = 1000000L;

  uint32_t start_clock;
  uint32_t next_clock;
}; // InOpSy

#endif