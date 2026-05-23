import urllib.request
import os

def download_file(url, target_path):
    print(f"Downloading {url} to {target_path}...")
    try:
        # Use a user agent to avoid being blocked or redirected to HTML
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            content = response.read()
            
            # Check if it's HTML (indicating a 404 page or blob view)
            if content.strip().startswith(b"<!DOCTYPE html>") or b"<html" in content[:200]:
                print(f"[Error] Downloaded content for {target_path} appears to be HTML, not raw data.")
                return False
            
            with open(target_path, 'wb') as f:
                f.write(content)
            print(f"Successfully downloaded {target_path} ({len(content)} bytes)")
            return True
    except Exception as e:
        print(f"[Error] Failed to download {url}: {e}")
        return False

def main():
    base_url = "https://raw.githubusercontent.com/opencv/opencv_3rdparty/wechat_qrcode_20210119/"
    files = ["detect.prototxt", "detect.caffemodel", "sr.prototxt", "sr.caffemodel"]
    
    os.makedirs("models/wechat_barcode", exist_ok=True)
    
    success_count = 0
    for f in files:
        url = base_url + f
        target = os.path.join("models/wechat_barcode", f)
        if download_file(url, target):
            success_count += 1
            
    if success_count == len(files):
        print("\n[Success] All WeChat QR model files downloaded successfully.")
    else:
        print(f"\n[Partial Failure] Only {success_count}/{len(files)} files downloaded correctly.")

if __name__ == "__main__":
    main()
