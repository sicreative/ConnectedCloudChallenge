#include "mbed.h"

#define NTP_DEFULT_NIST_SERVER_ADDRESS    "2.pool.ntp.org"
#define NTP_DEFULT_NIST_SERVER_PORT       123
#define NTP_SEND_BUF_LEN                  12
#define NTP_RCV_BUF_LEN                   12
#define NTP_RECEIVED_PARTIAL_DATA         -1
#define NTP_NETWORK_INTERFACE_UNAVAILABLE -2

class NTPClient {
    public:
        explicit NTPClient(NetworkInterface *interface = NULL);
        void set_server(char* server, int port);
        time_t get_timestamp(int timeout_ms = 15000);
        void network(NetworkInterface *interface);

    private:
        NetworkInterface *_iface;
        char *_nist_server_address;
        int _nist_server_port;

        uint32_t _ntohl(uint32_t num);
};
