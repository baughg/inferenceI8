#include "PerformanceCounter.h"

using namespace GB;

PerformanceCounter::PerformanceCounter()
	: MsTimer()
{
}


PerformanceCounter::~PerformanceCounter()
{
}

uint64_t PerformanceCounter::start()
{
	unsigned int ui = 0;	
	start_time_ = get_time_ms();
	start_clock_tick_ = __rdtscp(&ui);
	return start_time_;
}

uint64_t PerformanceCounter::stop()
{
	unsigned int ui = 0;
	stop_clock_tick_ = __rdtscp(&ui);
	end_time_ = get_time_ms();
	return end_time_;
}

std::ostream& GB::operator<<(std::ostream& os, const PerformanceCounter&counter)
{
	const auto ticks{ counter.stop_clock_tick_ - counter.start_clock_tick_ };
	const auto elapsed_time{ counter.elapsed() };
	double freq{ static_cast<double>(ticks) / (1.0e-3 * static_cast<double>(elapsed_time)) };
	freq /= 1.0e6;

	os << elapsed_time << "ms, ticks: " << ticks << " freq: " << freq << "MHz" << '\n';
	return os;
}