import argparse
import os

def convert_pdf_to_markdown(pdf_path, output_path=None):
    try:
        from marker.converters.pdf import PdfConverter
        from marker.models import create_model_dict
    except ImportError:
        print("Make sure Marker is installed!")
        return False
    
    if not os.path.isfile(pdf_path):
        print(f"'{pdf_path}' not found.")
        return False
    
    try:
        converter = PdfConverter(
            artifact_dict=create_model_dict(),
        )
        
        print(f"Converting {pdf_path}...")

        rendered = converter(pdf_path)

        if output_path is None:
            output_dir = "conversions"
            output_filename = os.path.splitext(os.path.basename(pdf_path))[0] + ".md"
        else:
            if os.path.splitext(output_path)[1]: 
                output_dir, output_filename = os.path.split(output_path)
            else:
                output_dir = output_path 
                output_filename = os.path.splitext(os.path.basename(pdf_path))[0] + ".md"

        os.makedirs(output_dir, exist_ok=True)

        final_output_path = os.path.join(output_dir, output_filename)

        with open(final_output_path, 'w', encoding='utf-8') as f:
            f.write(rendered.markdown)
        
        print(f"Complete!")
        return True
    
    except Exception as e:
        import traceback
        print(f"Error: {str(e)}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf_path")
    parser.add_argument("--output")
    
    args = parser.parse_args()
    
    convert_pdf_to_markdown(args.pdf_path, args.output)