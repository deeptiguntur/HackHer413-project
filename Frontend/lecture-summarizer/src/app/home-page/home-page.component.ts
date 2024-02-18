import { Component } from '@angular/core';
import { AppService } from '../app.service';

@Component({
  selector: 'app-home-page',
  templateUrl: './home-page.component.html',
  styleUrl: './home-page.component.scss'
})
export class HomePageComponent {
  urlVideo: any = null;
  urlPDF: any = null;
  format: any;
  spinner = false;
  summary = null;
  
  constructor(private appService: AppService) {}

  onSelectFile(event: any) {
    const file = event.target.files && event.target.files[0];
    if (file) {
      var reader = new FileReader();
      reader.readAsDataURL(file);
      if(file.type.indexOf('image')> -1){
        this.format = 'image';
      } else if(file.type.indexOf('video')> -1){
        this.format = 'video';
      }
      reader.onload = (event) => {
        this.urlVideo = (<FileReader>event.target).result;
        console.log(this.urlVideo)
      }
    }
  }

  pdfUpload(event: any) {
    const file = event.target.files && event.target.files[0];
    if (file) {
      var reader = new FileReader();
      reader.readAsDataURL(file);
      reader.onload = (event) => {
        this.urlPDF = (<FileReader>event.target).result;
        console.log(this.urlPDF)
      }
    }
  }

  generateText() {
    const data = {
      "video": this.urlVideo ? this.urlVideo : null,
      "PDF": this.urlPDF ? this.urlPDF : null
    }
    this.spinner = true;
    this.summary = null;
    this.appService.uploadData(data).subscribe(data => {
      console.log(data);
      this.summary = data.msg;
      this.spinner = false;
    }, (error) => {
      this.spinner = false;
    })
  }
}
