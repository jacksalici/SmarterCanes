#ifndef ANIMATIONS_H
#define ANIMATIONS_H
#include <Arduino.h>

const uint32_t stickman_walking[][4] = {
	{
		0x4006061b,
		0xf047f820,
		0x0,
		85
	},
	{
		0x40679,
		0xf067f820,
		0x0,
		66
	},
	{
		0x600,
		0xfff7e020,
		0x0,
		66
	},
	{
		0x0,
		0x6fff087f,
		0x2000000,
		85
	},
	{
		0x6,
		0x6f9f0871,
		0x2e00000,
		85
	},
	{
		0xe7,
		0x618f087f,
		0x2000000,
		85
	},
	{
		0xc0037,
		0xe187f870,
		0x2000000,
		85
	},
	{
		0x40060637,
		0xf087f820,
		0x0,
		66
	}
};

#endif // STICKMAN_WALKING_H