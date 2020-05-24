#include "NTPClient.h"
#include "mbed.h"

NTPClient::NTPClient(NetworkInterface *interface)
    : _iface(interface), _nist_server_address((char *)NTP_DEFULT_NIST_SERVER_ADDRESS), _nist_server_port(NTP_DEFULT_NIST_SERVER_PORT) {
}

void NTPClient::set_server(char* server, int port) {
    _nist_server_address = server;
    _nist_server_port = port;
}

time_t NTPClient::get_timestamp(int timeout_ms) {
    const time_t TIME1970 = (time_t)2208988800UL;
    int ntp_send_values[NTP_SEND_BUF_LEN] = {0};
    int ntp_recv_values[NTP_RCV_BUF_LEN] =  {0};

    SocketAddress nist;

    if (_iface) {
        int ret_gethostbyname = _iface->gethostbyname(_nist_server_address, &nist);




        if (ret_gethostbyname < 0) {
            // Network error on DNS lookup Try IP


        		return ret_gethostbyname;


        }

        nist.set_port(_nist_server_port);


        memset(ntp_send_values, 0x00, sizeof(ntp_send_values));
        ntp_send_values[0] = '\x1b';

        memset(ntp_recv_values, 0x00, sizeof(ntp_recv_values));

        UDPSocket sock;
        sock.open(_iface);
        sock.set_timeout(timeout_ms);

        sock.sendto(nist, (void*)ntp_send_values, sizeof(ntp_send_values));

        SocketAddress source;
        const int n = sock.recvfrom(&source, (void*)ntp_recv_values, sizeof(ntp_recv_values));

        if (n > 10) {
            return _ntohl(ntp_recv_values[10]) - TIME1970;

        } else {
            if (n < 0) {
                // Network error
                return n;

            } else {
                // No or partial data returned
                return NTP_RECEIVED_PARTIAL_DATA;
            }
        }

    } else {
        // No network interface
        return NTP_NETWORK_INTERFACE_UNAVAILABLE;
    }
}

void NTPClient::network(NetworkInterface *interface) {
    _iface = interface;
}

/*
 * ntohl: network to host long
 * TCP/IP standard Network Byte Order is Big-Endian.
 * So the Little-endian platforms need to convert the network byte
 * order (big-endian) to its native byte order.
 */
uint32_t NTPClient::_ntohl(uint32_t x) {
    uint32_t ret = (x & 0xff) << 24;
    ret |= (x & 0xff00) << 8;
    ret |= (x & 0xff0000UL) >> 8;
    ret |= (x & 0xff000000UL) >> 24;
    return ret;
}
