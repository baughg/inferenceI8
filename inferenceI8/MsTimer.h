//--------------------------------------------
//File Name   : ms_timer
//Description : milli-second timer for measuring emulation runtime
//Author      : Gary Baugh (baughg@tcd.ie)
//Date        : 12.05.2017
//--------------------------------------------

#pragma once
#include <stdint.h>

namespace GB {
	class MsTimer
	{
	public:
		MsTimer();
		~MsTimer();
		uint64_t get_time_ms();
		virtual uint64_t start();
		virtual uint64_t stop();
		int64_t elapsed() const;
	protected:
		uint64_t start_time_;
		uint64_t end_time_;
	};
}
