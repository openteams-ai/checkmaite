#!/bin/bash
# Prepare an Ubuntu product filesystem after docker/snapshot_apt.sh has run.
set -euo pipefail

# Remove apt, its signing chain, tar, and dpkg itself in one dpkg call (dpkg
# refuses to start once tar is gone). Nothing in the image needs them at run
# time. /var/lib/dpkg/status is kept so scanners still see every package.
dpkg --purge --force-remove-essential --force-depends \
    apt libapt-pkg6.0t64 gpg gpgv gpgconf ubuntu-keyring tar dpkg
rm -rf /etc/apt /var/lib/apt /var/cache/apt /var/cache/debconf

# Drop documentation but keep every package's copyright file: they carry the
# licence terms that redistribution requires and that SBOM tools read.
find /usr/share/doc -mindepth 1 ! -name copyright ! -type d -delete
find /usr/share/doc -mindepth 1 -type d -empty -delete
rm -rf \
    /var/cache/* \
    /var/log/* \
    /usr/share/man \
    /usr/share/info \
    /usr/share/python-wheels \
    /usr/lib/python3.12/ensurepip

printf 'checkmaite:x:10001:10001::/cache:/usr/sbin/nologin\n' >> /etc/passwd
printf 'checkmaite:x:10001:\n' >> /etc/group
mkdir -p /checkmaite /output /cache
chown -R 10001:10001 /output /cache
find / -xdev -type f \( -perm -4000 -o -perm -2000 \) -exec chmod a-s {} +

for command in apt apt-get dpkg gpg gpgv tar; do
    test ! -e "/usr/bin/${command}"
done
test -x /usr/sbin/nologin
/usr/bin/python3.12 -c \
    "import importlib.util; assert importlib.util.find_spec('ensurepip') is None"
test -s /var/lib/dpkg/status
test -n "$(find /usr/share/doc -name copyright -print -quit)"
test -z "$(find / -xdev -type f \( -perm -4000 -o -perm -2000 \) -print -quit)"
