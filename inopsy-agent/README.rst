Inopsy-agent
======

Agent is based on ngtcp2 implementation of QUIC: https://github.com/ngtcp2/ngtcp2.

Requirements
------------

The libngtcp2 C library itself does not depend on any external
libraries.  The example client, and server are written in C++17, and
should compile with the modern C++ compilers (e.g., clang >= 8.0, or
gcc >= 8.0).

The following packages are required to configure the build system:

- pkg-config >= 0.20
- autoconf
- automake
- autotools-dev
- libtool

libngtcp2 uses cunit for its unit test frame work:

- cunit >= 2.1

To build sources under the examples directory, libev and nghttp3 are
required:

- libev
- `nghttp3 <https://github.com/ngtcp2/nghttp3>`_ for HTTP/3

ngtcp2 crypto helper library, and client and server under examples
directory require at least one of the following TLS backends:

- `OpenSSL with QUIC support
  <https://github.com/quictls/openssl/tree/OpenSSL_1_1_1m+quic>`_
- GnuTLS >= 3.7.2
- BoringSSL (commit 36a41bf0bf2dd3176f8780e09c03585351f29963)
- Picotls (commit 821997cb35ecf02d4518a1b5749a3cd6200b5b87)

Install dependencies (Ubuntu version)
--------------------

.. code-block:: text

   $ sudo apt install gcc g++ git cmake libev-dev pkg-config autoconf automake autotools-dev libtool \
                      libcunit1 libcunit1-doc libcunit1-dev

Build from git
--------------

.. code-block:: text

   $ git clone --depth 1 -b OpenSSL_1_1_1m+quic https://github.com/quictls/openssl
   $ cd openssl
   $ # For Linux
   $ ./config enable-tls1_3 --prefix=$PWD/build
   $ make -j$(nproc)
   $ make install_sw
   $ cd ..
   $ git clone https://github.com/ngtcp2/nghttp3
   $ cd nghttp3
   $ # We need specific commit for nghttp3
   $ git checkout -b inopsy-agent-state 6faaea03b0fb20c295aea3016e3eaaf1bce3f0eb
   $ autoreconf -i
   $ ./configure --prefix=$PWD/build --enable-lib-only
   $ make -j$(nproc) check
   $ make install
   $ cd ..
   $ # Install inopsy-agent (temporary ngtcp2: https://github.com/ngtcp2/ngtcp2)
   $ git clone ssh://git@phabricator.arccn.ru:2222/diffusion/34/inopsy-agent.git
   $ cd inopsy-agent # cd ngtcp2
   $ # Temporary solution for development process
   $ git checkout develop
   $ autoreconf -i
   $ # For Mac users who have installed libev with MacPorts, append
   $ # ',-L/opt/local/lib' to LDFLAGS, and also pass
   $ # CPPFLAGS="-I/opt/local/include" to ./configure.
   $ # For OpenSSL >= v3.0.0, replace "openssl/build/lib" with
   $ # "openssl/build/lib64".
   $ ./configure PKG_CONFIG_PATH=$PWD/../openssl/build/lib/pkgconfig:$PWD/../nghttp3/build/lib/pkgconfig LDFLAGS="-Wl,-rpath,$PWD/../openssl/build/lib"
   $ make -j$(nproc) check

Client/Server
-------------

After successful build, the client and server executable should be
found under examples directory.  They talk HTTP/3.

To test if building is successful you can launch server and client.

Server
~~~~~~

.. code-block:: text

   $ examples/server [OPTIONS] <ADDR> <PORT> <PRIVATE_KEY_FILE> <CERTIFICATE_FILE>

The notable options are:

- ``-V``, ``--validate-addr``: Enforce stateless address validation.

Client
~~~~~~

.. code-block:: text

   $ examples/client [OPTIONS] <HOST> <PORT> [<URI>...]

The notable options are:

- ``-d``, ``--data=<PATH>``: Read data from <PATH> and send it to a
  peer.


Configuring Wireshark for QUIC
------------------------------

`Wireshark <https://www.wireshark.org/download.html>`_ can be configured to
analyze QUIC traffic using the following steps:

1. Set *SSLKEYLOGFILE* environment variable:

   .. code-block:: text

      $ export SSLKEYLOGFILE=quic_keylog_file

2. Set the port that QUIC uses

   Go to *Preferences->Protocols->QUIC* and set the port the program
   listens to.  In the case of the example application this would be
   the port specified on the command line.

3. Set Pre-Master-Secret logfile

   Go to *Preferences->Protocols->TLS* add set the *Pre-Master-Secret
   log file* to the same value that was specified for *SSLKEYLOGFILE*.

4. Choose the correct network interface for capturing

   Make sure you choose the correct network interface for
   capturing. For example, if using localhost choose the *loopback*
   network interface on macos.

5. Create a filter

   Create A filter for the udp.port and set the port to the port the
   application is listening to. For example:

   .. code-block:: text

      udp.port == 7777
